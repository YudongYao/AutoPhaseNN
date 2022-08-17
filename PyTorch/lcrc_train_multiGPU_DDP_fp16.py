from __future__ import print_function
import os
import glob
import argparse
import time
import socket
import numpy as np
import torch
from torch import nn
from torch import optim
import sys
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import json
from matplotlib import pyplot as plt

from AutoPhaseNN_model import Network
from data_loader2 import *


def metric_average(val, name, with_ddp, world_size):
    if (with_ddp):
        # Sum everything and divide by total size:
        # dist.all_reduce(val,op=dist.reduce_op.SUM)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        val /= world_size
    else:
        pass
    return val


def train(args, model, criterion, trainloader, optimizer, scheduler, epoch, with_DDP, rank, scaler):

    # Training
    model.train()

    start_time = time.time()
    loss_ft = torch.tensor(0.0)
    loss_amp = torch.tensor(0.0)
    loss_ph = torch.tensor(0.0)

    if args.device == 'cuda':
        loss_ft = loss_ft.cuda()
        loss_amp = loss_amp.cuda()
        loss_ph = loss_ph.cuda()

    for i, (ft_images, amps, phs) in tqdm(enumerate(trainloader)):
        # Transfer to GPU
        if args.device == 'cuda':
            ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            y, _, pred_amps, pred_phs, support = model(
                ft_images)  # Forward pass

            # Compute losses
            loss_f = criterion(y, ft_images)
            loss_a = criterion(pred_amps, amps)  # Monitor amplitude loss
            # Monitor phase loss but only within support (which may not be same as true amp)
            loss_p = criterion(pred_phs*support, phs*support)
    #
            if args.unsupervise:
                loss = loss_f  # Use only FT loss for gradients
            else:
                loss = loss_a + loss_p + loss_f

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_ft += loss_f.item()
        loss_amp += loss_a.item()
        loss_ph += loss_p.item()

        if args.lr_type == 'clr':
            # Update the LR according to the schedule -- CyclicLR updates each batch
            scheduler.step()

    loss_ft /= i
    loss_amp /= i
    loss_ph /= i

    loss_ft_avg = metric_average(
        loss_ft, 'train_loss_ft', with_DDP, args.world_size)
    loss_amp_avg = metric_average(
        loss_amp, 'train_loss_amp', with_DDP, args.world_size)
    loss_ph_avg = metric_average(
        loss_ph, 'train_loss_ph', with_DDP, args.world_size)

    time_cost = time.time() - start_time

    if args.global_rank == 0:

        print('\nTraining epoch: {}, loss_ft: {:.4e}, loss_amp: {:.4e}, loss_ph: {:.4e}'.format(
            epoch, loss_ft_avg, loss_amp_avg, loss_ph_avg, time_cost))

    return loss_ft_avg, loss_amp_avg, loss_ph_avg

def validation(args, model, criterion, validloader, optimizer, epoch, with_DDP, rank):
    model.eval()

    val_loss_ft = torch.tensor(0.0)
    val_loss_amp = torch.tensor(0.0)
    val_loss_ph = torch.tensor(0.0)

    if args.device == 'cuda':
        val_loss_ft, val_loss_amp, val_loss_ph = val_loss_ft.cuda(
        ), val_loss_amp.cuda(), val_loss_ph.cuda()

    with torch.no_grad():
        for j, (ft_images, amps, phs) in enumerate(validloader):

            if args.device == 'cuda':
                ft_images, amps, phs = ft_images.cuda(), amps.cuda(), phs.cuda()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                y, _, pred_amps, pred_phs, support = model(
                    ft_images)  # Forward pass
                # Compute losses
                loss_f = criterion(y, ft_images)
                loss_a = criterion(pred_amps, amps)  # Monitor amplitude loss
                # Monitor phase loss but only within support (which may not be same as true amp)
                loss_p = criterion(pred_phs*support, phs*support)

            val_loss_ft += loss_f.item()
            val_loss_amp += loss_a.item()
            val_loss_ph += loss_p.item()

    val_loss_ft /= j
    val_loss_amp /= j
    val_loss_ph /= j

    val_loss_ft = metric_average(
        val_loss_ft, 'validation_loss_ft', with_DDP, args.world_size)
    val_loss_amp = metric_average(
        val_loss_amp, 'validation_loss_amp', with_DDP, args.world_size)
    val_loss_ph = metric_average(
        val_loss_ph, 'validation_loss_ph', with_DDP, args.world_size)

    if args.global_rank == 0:
        print('\nTraining epoch: {}, val_loss_ft: {:.4e}, val_loss_amp: {:.4e}, val_loss_ph: {:.4e}'.format(
            epoch, val_loss_ft, val_loss_amp, val_loss_ph))

    return val_loss_ft, val_loss_amp, val_loss_ph

if __name__ == "__main__":

    print('Starting script \n pytorch version: {}'.format(
        torch.__version__))
    # torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    t0 = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--OutputFolder',
        type=str,
        default='/lcrc/project/AutoPhase/test_pytorch/'
    )
    parser.add_argument('--unsupervise', action='store_true', default=False)
    # dataset
    parser.add_argument('--DataFolder', type=str,
                        default='/lcrc/project/AutoPhase/CDI_simulation_upsamp_aug_220429/')
    parser.add_argument('--DataSummary',
                        type=str, default='3D_upsamp.txt')
    parser.add_argument('--batch_size', default=16,
                        type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='training epochs')
    parser.add_argument('--train_size',
                        type=int,
                        default=60000,
                        help='training data size')
    parser.add_argument('--train_perc', type=float, default=0.9)
    parser.add_argument('--loss_type',
                        type=str,
                        default='mae',
                        help='loss type')
    parser.add_argument('--Initlr',
                        type=float,
                        default=1e-3,
                        help='initial lr')
    parser.add_argument('--lr_type', type=str, default='clr', help='lr type')
    parser.add_argument('--optim_type', type=str,
                        default='adam', help='lr optim_type')
    parser.add_argument('--shape',
                        default=64,
                        type=int,
                        help='input data size')
    parser.add_argument('--use_down_stride',
                        action='store_true', default=False)
    parser.add_argument('--use_up_stride', action='store_true', default=False)
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--nconv', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--scale_I', type=int, default=0)
    parser.add_argument('--num_workers', default=4,
                        type=int, help='num of workers')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        metavar='N', help='Local process rank.')
    parser.add_argument('--gpu_device', default=None, type=int)

    parser.add_argument('--notes', type=str, default='test')

    args = parser.parse_args()

    arguments_strOut = args.OutputFolder
    if not os.path.exists(arguments_strOut):
        os.makedirs(arguments_strOut)

    total_train_size = args.train_size
    batch_size = args.batch_size
    DataSummary = args.DataSummary
    data_path = args.DataFolder
    result_path = args.OutputFolder
    scale_I = args.scale_I

    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    # print(args.__dict__)
    print('use device: {}'.format(args.device))

    # DDP: initialize library.
    '''Initialize distributed communication'''

    # What backend?  nccl on GPU, gloo on CPU
    if args.device == "cuda":
        backend = 'nccl'
    elif args.device == "cpu":
        backend = 'gloo'

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1

    if args.world_size > 1:
        print('have {} gpus, use DDP'.format(args.world_size))
        print('visiable gpus: {}'.format(torch.cuda.device_count()))
        with_ddp = True
    else:
        with_ddp = False

    if with_ddp:
        print('local rank before change: {}'.format(args.local_rank))
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu_device = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu_device = args.rank % torch.cuda.device_count()

        torch.distributed.init_process_group(
            backend=backend, init_method='env://')

        torch.cuda.set_device(args.gpu_device)
        args.global_rank = dist.get_rank()
        print('global rank: {}'.format(dist.get_rank()))

    if args.device == 'cuda':
        # DDP: pin GPU to local rank.
        torch.cuda.manual_seed(args.seed)

    if (args.num_threads != 0):
        torch.set_num_threads(args.num_threads)

    if args.rank == 0:
        print("Torch Thread setup: ")
        print(" Number of threads: ", torch.get_num_threads())

    kwargs = {'num_workers': args.num_workers,
              'pin_memory': True} if args.device.find("cuda") != -1 else {}

    if args.global_rank == 0:
        with open(os.path.join(args.OutputFolder, 'setting.json'), 'w') as f:
            f.write(json.dumps(args.__dict__, indent=4))

    args.device_type = torch.device(args.device)

    if args.global_rank == 0:
        layout = {
            "": {
                "Loss_ft": ["Multiline", ["Loss_ft/train", "Loss_ft/validation"]],
                "Loss_amp": ["Multiline", ["Loss_amp/train", "Loss_amp/validation"]],
                "Loss_ph": ["Multiline", ["Loss_ph/train", "Loss_ph/validation"]],
                'LR': ["Multiline", ["Lr"]]
            },
        }
        writer = SummaryWriter(comment=os.path.basename(
            os.path.dirname(result_path)))
        writer.add_custom_scalars(layout)

    model = Network(args)

    if args.device == 'cuda':
        if args.gpu_device is not None:
            # Move model to GPU.
            # model.cuda()
            model.cuda(args.gpu_device)
        else:
            model.cuda()

    print('model parameters: {}'.format(
        sum([param.nelement() for param in model.parameters()])))

    # load data
    
    dataname_list = os.path.join(data_path, DataSummary)
    filelist = []

    with open(dataname_list, 'r') as f:
        txtfile = f.readlines()
    for i in range(len(txtfile)):
        tmp = str(txtfile[i]).split('/')[-1]
        tmp = tmp.split('\n')[0]

        filelist.append(tmp)
    f.close()
    print('number of available file:%d' % len(filelist))

    # give training data size and filelist
    train_filelist = filelist[:total_train_size]
    print('number of training:%d' % len(train_filelist))

    train_dataset = Dataset(train_filelist, data_path, load_all=False,
                            ratio=args.train_perc, dataset='train', scale_I=scale_I)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, **kwargs)

    validation_dataset = Dataset(
        train_filelist, data_path, load_all=False, ratio=args.train_perc, dataset='validation', scale_I=scale_I)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset, num_replicas=args.world_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=validation_sampler, shuffle=False, **kwargs)

    # Setup optimizer and learning rate
    # Optimizer details
    LR = args.Initlr * args.world_size
    # set loss function
    criterion = nn.L1Loss()

    # set optimizer
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # set learning rate scheduler
    if args.lr_type == 'clr':
        iterations_per_epoch = np.floor(
            (total_train_size*args.train_perc)/(batch_size*args.world_size))+1
        # Paper recommends 2-10 number of iterations, step_size is half cycle
        step_size = 6*iterations_per_epoch
        print("LR step size is:", step_size, "which is every %d epochs" %
              (step_size/iterations_per_epoch))
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10,
                                                      max_lr=LR, step_size_up=step_size,
                                                      cycle_momentum=False, mode='triangular2')
    elif args.lr_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)
    elif args.lr_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    training_losses_ft = []
    training_losses_amp = []
    training_losses_ph = []
    validation_losses_ft = []
    validation_losses_amp = []
    validation_losses_ph = []
    epochs = 0

    if with_ddp:
        if args.gpu_device is not None:
            # wrap the model in DDP:
            model = DDP(model, device_ids=[
                        args.gpu_device], output_device=args.gpu_device, find_unused_parameters=False)
            # model = DDP(model, device_ids=[args.gpu_device], output_device=args.gpu_device)
        else:
            model = DDP(model)

    t_start = time.time()

    for epoch in range(epochs + 1, args.epoch+1):
        if args.distributed:

            train_loader.sampler.set_epoch(epoch)
            validation_loader.sampler.set_epoch(epoch)

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        loss_ft, loss_amp, loss_ph = train(args, model, criterion, train_loader, optimizer, scheduler,
                                        epoch, with_ddp, args.gpu_device, scaler)

        training_losses_ft.append(loss_ft)
        training_losses_amp.append(loss_amp)
        training_losses_ph.append(loss_ph)

        loss_ft, loss_amp, loss_ph = validation(args, model, criterion, validation_loader,
                                                optimizer, epoch, with_ddp, args.gpu_device)
        
        validation_losses_ft.append(loss_ft)
        validation_losses_amp.append(loss_amp)
        validation_losses_ph.append(loss_ph)

        if args.lr_type == 'step':
            scheduler.step()
        elif args.lr_type == 'Plateau':
            if args.unsupervise:
                val_loss = validation_losses_ft
            else:
                val_loss = validation_losses_ft+validation_losses_amp+validation_losses_ph
            scheduler.step(val_loss)

        if epoch % args.save_model == 0:
            if args.global_rank == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if with_ddp else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'training_losses_ft': training_losses_ft,
                        'training_losses_amp': training_losses_amp,
                        'training_losses_ph': training_losses_ph,
                        'validation_losses_ft': validation_losses_ft,
                        'validation_losses_amp': validation_losses_amp,
                        'validation_losses_ph': validation_losses_ph,
                    }, arguments_strOut + '/training_model_{:06d}'.format(epoch) + '.pt')
                print('checkpoint saved!')

        if args.global_rank == 0:
            writer.add_scalar("Loss_ft/train", training_losses_ft[-1], epoch)
            writer.add_scalar("Loss_ft/validation",
                              validation_losses_ft[-1], epoch)
            writer.add_scalar("Loss_amp/train", training_losses_amp[-1], epoch)
            writer.add_scalar("Loss_amp/validation",
                              validation_losses_amp[-1], epoch)
            writer.add_scalar("Loss_ph/train", training_losses_ph[-1], epoch)
            writer.add_scalar("Loss_ph/validation",
                              validation_losses_ph[-1], epoch)
            writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch)

    t1 = time.time()
    if args.global_rank == 0:
        print("Total running time: %s seconds" % (t1 - t0))
        print("Average time per epoch: {:.2f}s".format(
            (t1-t_start)/(args.epoch-epochs)))

        writer.flush()
        writer.close()