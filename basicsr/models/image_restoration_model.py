import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
#from audtorch.metrics.functional import pearsonr
import torch.autograd as autograd

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.psnr_best = -1

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        #if train_opt.get('pixel_opt2'):
        #    pixel_type = train_opt['pixel_opt2'].pop('type')
        #    cri_pix_cls2 = getattr(loss_module, pixel_type)
        #    self.cri_pix2 = cri_pix_cls2(**train_opt['pixel_opt2']).to(
        #        self.device)
        #else:
        #    raise ValueError('pixel loss are None.')

        if train_opt.get('SSIM'):
            ssim_type = train_opt['SSIM'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['SSIM']).to(
                self.device)
        else:
            raise ValueError('ssim loss are None.')

        if train_opt.get('color_opt'):
            color_type = train_opt['color_opt'].pop('type')
            cri_color_cls = getattr(loss_module, color_type)
            self.cri_color = cri_color_cls(**train_opt['color_opt']).to(
                self.device)
        else:
            self.cri_color = None

        #if train_opt.get('VGG'):
        #    vgg_type = train_opt['VGG'].pop('type')
        #    cri_vgg_cls = getattr(loss_module, vgg_type)
        #    self.cri_vgg = cri_vgg_cls(**train_opt['VGG']).to(
        #        self.device)
        #else:
        #    raise ValueError('vgg loss are None.')

        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'label' in data:
            self.label = data['label']
#            self.label = torch.nn.functional.one_hot(data['label'], num_classes=3)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        elif hasattr(self, 'gt'):
            del self.gt

    def check_inf_nan(self, x):
        x[x.isnan()] = 0
        x[x.isinf()] = 1e7
        return x
    def compute_correlation_loss(self, x1, x2):
        b, c = x1.shape[0:2]
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)
#        print(x1, x2)
        pearson = (1. - self.cri_seq(x1, x2)) / 2.
        return pearson[~pearson.isnan()*~pearson.isinf()].mean()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, )

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        loss_dict['l_pix'] = l_pix

        #l2_pix = self.cri_pix2(self.output, self.gt)
        #loss_dict['l2_pix'] = l2_pix

        l_ssim = self.cri_ssim(self.output, self.gt)
        loss_dict['SSIM'] = l_ssim

        l_color = self.output.new_zeros(())
        if self.cri_color is not None:
            l_color = self.cri_color(self.output, self.gt)
            loss_dict['l_color'] = l_color

        #l_vgg = self.cri_vgg(self.output, self.gt)
        #loss_dict['VGG'] = l_vgg


        #print("ssim:", l_ssim, "pix:", l_pix, "vgg:", l_vgg)
        loss_total = l_pix + l_ssim + l_color
        if not torch.isfinite(loss_total):
            raise FloatingPointError(
                f'Non-finite training loss at iter {current_iter}: '
                f'{dict(loss_dict)}')
        loss_total.backward()
        
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01, error_if_nonfinite=False)
        self.optimizer_g.step()

        self.log_dict, self.loss_total = self.reduce_loss_dict(loss_dict)
        self.loss_dict = loss_dict
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = (
            self.opt['val'].get('metrics') is not None
            and not dataloader.dataset.opt.get('visual_only', False))
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        metric_cnt = 0
        max_samples = dataloader.dataset.opt.get('max_samples')

        for idx, val_data in enumerate(dataloader):
            if max_samples is not None and idx >= int(max_samples):
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if val_data.get('resized', False) and val_data.get('resize_back', True):
                ori_h = int(val_data['ori_height'][0])
                ori_w = int(val_data['ori_width'][0])
                sr_img = cv2.resize(sr_img, (ori_w, ori_h), interpolation=cv2.INTER_LANCZOS4)
            has_gt = 'gt' in visuals
            if has_gt:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             dataset_name,
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    '''
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')'''
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    '''save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')'''
                    
                imwrite(sr_img, save_img_path)
                #imwrite(gt_img, save_gt_img_path)

            if with_metrics and has_gt:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                metric_cnt += 1

        current_metric = 0.
        if with_metrics and metric_cnt > 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= metric_cnt
                current_metric = max(current_metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if metric == 'psnr' and value >= self.psnr_best:
                self.save(0, current_iter, best=True)
                self.psnr_best = value
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, best=False):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'], best=best)
        else:
            self.save_network(self.net_g, 'net_g', current_iter, best=best)
        self.save_training_state(epoch, current_iter, best=best)
