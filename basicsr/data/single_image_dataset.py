from os import path as osp
import numpy as np
from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir


class SingleImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.resize_long_side = int(opt.get('resize_long_side', 0))
        self.resize_back = opt.get('resize_back', True)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.lq_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        ori_h, ori_w = img_lq.shape[:2]
        img_lq, resized = self._resize_long_side(img_lq)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'lq_path': lq_path,
            'ori_height': ori_h,
            'ori_width': ori_w,
            'resized': resized,
            'resize_back': self.resize_back
        }

    def __len__(self):
        return len(self.paths)

    def _resize_long_side(self, img_lq):
        if self.resize_long_side <= 0:
            return img_lq, False

        h, w = img_lq.shape[:2]
        max_side = max(h, w)
        if max_side <= self.resize_long_side:
            return img_lq, False

        scale = self.resize_long_side / max_side
        new_size = (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))))

        # imfrombytes returns BGR float32. Resize with PIL LANCZOS to match
        # the standalone test script, then convert back to BGR float32.
        img_rgb = np.clip(img_lq[:, :, ::-1] * 255.0, 0, 255).round().astype(np.uint8)
        img_rgb = Image.fromarray(img_rgb).resize(new_size, Image.Resampling.LANCZOS)
        img_rgb = np.asarray(img_rgb).astype(np.float32) / 255.0
        return img_rgb[:, :, ::-1].copy(), True
