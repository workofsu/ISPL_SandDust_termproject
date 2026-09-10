"""Calculate PSNR and SSIM for restored/ground-truth image pairs."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from basicsr.metrics import calculate_psnr, calculate_ssim


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def index_images(folder):
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM.')
    parser.add_argument(
        '--config',
        default=str(PROJECT_ROOT / 'options' / 'sand_dust.yml'),
        help='Path to the single YAML configuration file.')
    args = parser.parse_args()

    with Path(args.config).resolve().open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    test_options = config['test']
    result_dir = resolve_path(test_options['result_dir'])
    target_dir = resolve_path(test_options['ground_truth_dir'])
    test_y_channel = bool(test_options.get('test_y_channel', True))
    if not result_dir.is_dir():
        raise FileNotFoundError(f'Result folder was not found: {result_dir}')
    if not target_dir.is_dir():
        raise FileNotFoundError(f'Ground-truth folder was not found: {target_dir}')

    results = index_images(result_dir)
    targets = index_images(target_dir)
    names = sorted(results.keys() & targets.keys())
    if not names:
        raise ValueError(
            'No result/ground-truth pairs with matching file names were found.')

    psnr_values = []
    ssim_values = []
    print('file, PSNR, SSIM')
    for name in names:
        result = cv2.imread(str(results[name]), cv2.IMREAD_COLOR)
        target = cv2.imread(str(targets[name]), cv2.IMREAD_COLOR)
        if result is None or target is None:
            raise ValueError(f'Failed to read image pair: {name}')
        if result.shape != target.shape:
            result = cv2.resize(
                result, (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_LANCZOS4)

        psnr = calculate_psnr(
            result, target, crop_border=0,
            input_order='HWC', test_y_channel=test_y_channel)
        ssim = calculate_ssim(
            result, target, crop_border=0,
            input_order='HWC', test_y_channel=test_y_channel)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        print(f'{name}, {psnr:.4f}, {ssim:.4f}')

    print(f'Pairs: {len(names)}')
    print(f'Evaluation channel: {"Y" if test_y_channel else "RGB"}')
    print(f'Average PSNR: {np.mean(psnr_values):.4f} dB')
    print(f'Average SSIM: {np.mean(ssim_values):.4f}')

    missing_results = targets.keys() - results.keys()
    missing_targets = results.keys() - targets.keys()
    if missing_results:
        print(f'Warning: {len(missing_results)} targets have no result image.')
    if missing_targets:
        print(f'Warning: {len(missing_targets)} results have no target image.')


if __name__ == '__main__':
    main()
