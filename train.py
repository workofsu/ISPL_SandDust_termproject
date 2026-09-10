"""Single-configuration training entry point for sand_dust model."""

import argparse
import datetime
import logging
import math
import os
import random
import time
from pathlib import Path
from os import path as osp

import torch
import numpy as np
import yaml
from tensorboardX import SummaryWriter

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs,
                           mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, normalize_options


PROJECT_ROOT = Path(__file__).resolve().parent


def initialize_runtime_options(opt, launcher='none'):
    """Add distributed-training and random-seed runtime state."""
    if launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if launcher == 'slurm' and 'dist_params' in opt:
            init_dist(launcher, **opt['dist_params'])
        else:
            init_dist(launcher)
            print('init dist .. ', launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def load_options(config_path, is_train=True, launcher='none', root_path=None):
    """Load the one user-facing YAML file and prepare BasicSR options."""
    with open(config_path, 'r', encoding='utf-8') as config_file:
        options = yaml.safe_load(config_file)
    if not isinstance(options, dict):
        raise ValueError(f'Invalid YAML configuration: {config_path}')
    opt = normalize_options(
        options,
        is_train=is_train,
        root_path=root_path,
    )
    return initialize_runtime_options(opt, launcher=launcher)


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    """Create the train dataloader and all validation dataloaders."""
    train_loader, val_loader, val_loaders = None, None, {}
    total_epochs, total_iters = 0, int(opt['train']['total_iter'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase.startswith('val'):
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

        if val_loader is not None:
            val_loaders[dataset_opt['name']] = val_loader
        val_loader = None

    if train_loader is None:
        raise ValueError('A train dataset must be defined in the YAML config.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def get_auto_resume_state(opt):
    """Find the latest saved training state for the current experiment."""
    state_folder_path = opt['path']['training_states']
    try:
        states = [
            file for file in os.listdir(state_folder_path)
            if file.endswith('.state') and file[:-6].isdigit()
        ]
    except FileNotFoundError:
        states = []

    if not states:
        return None

    max_state_file = f'{max(int(file[:-6]) for file in states)}.state'
    return os.path.join(state_folder_path, max_state_file)


def train_pipeline(opt):
    """Run one training job from a prepared option dictionary."""
    torch.backends.cudnn.benchmark = True

    # An explicitly configured state must take precedence over auto-resume.
    # This also allows recovery from a newer checkpoint containing NaN values.
    if not opt['path'].get('resume_state'):
        resume_state_path = get_auto_resume_state(opt)
        if resume_state_path is not None:
            opt['path']['resume_state'] = resume_state_path

    if opt['path'].get('resume_state'):
        if torch.cuda.is_available() and opt.get('num_gpu', 0) > 0:
            device_id = torch.cuda.current_device()
            map_location = lambda storage, loc: storage.cuda(device_id)
        else:
            map_location = 'cpu'
        resume_state = torch.load(
            opt['path']['resume_state'], map_location=map_location)
    else:
        resume_state = None

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)

    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}. '
                         "Supported ones are: None, 'cuda', 'cpu'.")

    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')
    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    scale = opt['scale']

    epoch = start_epoch
    loss_list = []
    loss_writer = SummaryWriter(opt['path']['log'])

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break

            if opt['train']['scheduler'].get('type') != 'ReduceLROnPlateau':
                model.update_learning_rate(
                    current_iter,
                    warmup_iter=opt['train'].get('warmup_iter', -1))
            else:
                if len(loss_list) >= 1000:
                    model.update_learning_rate(
                        current_iter,
                        warmup_iter=opt['train'].get('warmup_iter', -1),
                        value_scheduler=np.mean(loss_list))
                    loss_writer.add_scalar(
                        'loss sche_step', np.mean(loss_list), current_iter)
                    loss_list = []

            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info(
                    '\n Updating Patch_Size to {} and Batch_Size to {} \n'.
                    format(mini_gt_size,
                           mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']
            label = train_data['label']
            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size),
                                        k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]
                label = label[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]

            model.feed_train_data({'lq': lq, 'gt': gt, 'label': label})
            model.optimize_parameters(current_iter)
            for l_name in model.loss_dict:
                loss_writer.add_scalar(
                    f'{l_name} loss', model.loss_dict[l_name], current_iter)
            loss_list.append(model.loss_total)

            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (
                    current_iter % opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                for val_name, val_loader in val_loaders.items():
                    metric_out = model.validation(
                        val_loader, current_iter, tb_logger,
                        opt['val']['save_img'], rgb2bgr, use_image)
                    if metric_out != 0:
                        loss_writer.add_scalar(
                            f'psnr_{val_name}', metric_out, current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        epoch += 1

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)

    if opt.get('val') is not None:
        for val_loader in val_loaders.values():
            model.validation(val_loader, current_iter, tb_logger,
                             opt['val']['save_img'])

    loss_writer.close()
    if tb_logger:
        tb_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train sand_dust model.')
    parser.add_argument(
        '--config',
        default=str(PROJECT_ROOT / 'options' / 'sand_dust.yml'),
        help='Path to the single YAML configuration file.')
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    opt = load_options(
        Path(args.config).resolve(),
        launcher='none',
        root_path=str(PROJECT_ROOT),
    )
    train_pipeline(opt)


if __name__ == '__main__':
    main()
