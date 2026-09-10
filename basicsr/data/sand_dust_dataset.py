import os
from os import path as osp
import random

import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


DEFAULT_COLORS = [
    '#C89463', '#C6853D', '#C17137', '#BB9C87', '#B9A99C',
    '#B97455', '#B78E56', '#B56F4B', '#B37A43', '#B39163',
    '#A77135', '#A58961', '#A14A10', '#986339', '#977A38',
    '#8C6E38', '#84674C', '#826934', '#7D6E4A', '#7C766A',
    '#6F5633'
]


class Dataset_SandDustImage(data.Dataset):
    """Sand-dust paired image dataset.

    This dataset supports the sand-dust file pattern used by the reference
    dataloader, where one GT image can have multiple color-synthesized LQ
    images, for example:

        gt/0001.jpg
        lq/0001_#C89463.JPG
        lq/0001_#C6853D.JPG

    Important options:
        dataroot_gt (str): GT image folder.
        dataroot_lq (str): LQ/input image folder.
        colors (list[str]): Color suffixes used in LQ filenames.
        pairing_mode (str): 'all_colors' or 'one_color_per_gt'.
        filename_tmpl (str): Template before extension. Default '{}_{}'.
        gt_ext (str | list[str] | None): GT extensions to include.
        lq_ext (str): LQ extension appended when filename_tmpl has no ext.
        split (str): 'train', 'val', or 'all'. Default follows phase.
        val_ratio (float): GT-level validation ratio. Default 0.1.
        split_seed (int): GT-level train/val split seed. Default 1143.
        resize_gt_to_lq (bool): Resize GT to match LQ size when needed.
        strict_pairing (bool): Raise if expected LQ files are missing.
    """

    def __init__(self, opt):
        super(Dataset_SandDustImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.colors = opt.get('colors', DEFAULT_COLORS)
        self.filename_tmpl = opt.get('filename_tmpl', '{}_{}')
        self.gt_ext = opt.get('gt_ext', None)
        self.lq_ext = opt.get('lq_ext', '.JPG')
        self.pairing_mode = opt.get('pairing_mode', 'all_colors')
        self.strict_pairing = opt.get('strict_pairing', True)
        self.resize_gt_to_lq = opt.get('resize_gt_to_lq', True)
        self.val_ratio = float(opt.get('val_ratio', 0.1))
        self.split_seed = int(opt.get('split_seed', 1143))
        self.split = opt.get(
            'split', 'train' if opt.get('phase') == 'train' else 'val')

        if self.pairing_mode not in ('all_colors', 'one_color_per_gt'):
            raise ValueError(
                "pairing_mode must be 'all_colors' or 'one_color_per_gt', "
                f'but got {self.pairing_mode}.')
        if self.split not in ('train', 'val', 'all'):
            raise ValueError(
                "split must be 'train', 'val', or 'all', "
                f'but got {self.split}.')
        if not self.colors:
            raise ValueError('colors must contain at least one color value.')

        self.paths = self._make_paths()
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def _make_paths(self):
        gt_paths = self._scan_gt_paths()
        rng = random.Random(self.split_seed)
        rng.shuffle(gt_paths)

        split_idx = int(len(gt_paths) * (1 - self.val_ratio))
        indexed_paths = list(enumerate(gt_paths))
        if self.split == 'train':
            selected_paths = indexed_paths[:split_idx]
        elif self.split == 'val':
            selected_paths = indexed_paths[split_idx:]
        else:
            selected_paths = indexed_paths

        paths = []
        missing_paths = []
        for gt_index, gt_path in selected_paths:
            basename = osp.splitext(osp.basename(gt_path))[0]
            if self.pairing_mode == 'all_colors':
                color_items = list(enumerate(self.colors))
            else:
                color_idx = gt_index % len(self.colors)
                color_items = [(color_idx, self.colors[color_idx])]

            for color_idx, color in color_items:
                lq_path = osp.join(
                    self.lq_folder, self._format_lq_name(basename, color))
                if not osp.isfile(lq_path):
                    missing_paths.append(lq_path)
                    if not self.strict_pairing:
                        continue
                paths.append({
                    'lq_path': lq_path,
                    'gt_path': gt_path,
                    'label': color_idx,
                    'color': color
                })

        if missing_paths and self.strict_pairing:
            sample = '\n'.join(missing_paths[:10])
            raise FileNotFoundError(
                'Missing expected sand-dust LQ files. First missing files:\n'
                f'{sample}')
        if not paths:
            raise ValueError(
                f'No sand-dust pairs found for split={self.split}, '
                f'pairing_mode={self.pairing_mode}.')
        return paths

    def _scan_gt_paths(self):
        if not osp.isdir(self.gt_folder):
            raise FileNotFoundError(f'GT folder does not exist: {self.gt_folder}')

        extensions = self._normalize_extensions(self.gt_ext)
        paths = []
        for name in os.listdir(self.gt_folder):
            full_path = osp.join(self.gt_folder, name)
            if not osp.isfile(full_path):
                continue
            if extensions is None or osp.splitext(name)[1].lower() in extensions:
                paths.append(full_path)
        paths.sort()

        if not paths:
            raise ValueError(f'No GT images found in {self.gt_folder}.')
        return paths

    @staticmethod
    def _normalize_extensions(ext):
        if ext is None:
            return {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if isinstance(ext, str):
            ext = [ext]
        return {item.lower() if item.startswith('.') else f'.{item.lower()}'
                for item in ext}

    def _format_lq_name(self, basename, color):
        lq_name = self.filename_tmpl.format(basename, color)
        if osp.splitext(lq_name)[1]:
            return lq_name
        return f'{lq_name}{self.lq_ext}'

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        path_info = self.paths[index]

        gt_path = path_info['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise Exception(f'gt path {gt_path} not working') from exc

        lq_path = path_info['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception as exc:
            raise Exception(f'lq path {lq_path} not working') from exc

        img_gt = self._align_gt_to_lq(img_gt, img_lq, scale, gt_path, lq_path)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, gt_path)

            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'label': path_info['label']
        }

    def __len__(self):
        return len(self.paths)

    def _align_gt_to_lq(self, img_gt, img_lq, scale, gt_path, lq_path):
        h_lq, w_lq = img_lq.shape[:2]
        expected_gt_size = (w_lq * scale, h_lq * scale)
        h_gt, w_gt = img_gt.shape[:2]

        if (w_gt, h_gt) == expected_gt_size:
            return img_gt
        if not self.resize_gt_to_lq:
            raise ValueError(
                f'GT/LQ size mismatch. GT ({h_gt}, {w_gt}) is not {scale}x '
                f'LQ ({h_lq}, {w_lq}). gt_path={gt_path}, lq_path={lq_path}')

        return cv2.resize(img_gt, expected_gt_size, interpolation=cv2.INTER_AREA)
