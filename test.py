"""Direct folder inference for sand_dust model."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from basicsr.models.archs.sand_dust_model_arch import SandDustModel


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def load_config(path):
    with path.open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'Invalid YAML configuration: {path}')
    return config


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(config, device):
    network_options = config['network_g'].copy()
    network_type = network_options.pop('type')
    if network_type != 'SandDustModel':
        raise ValueError(
            f'Only SandDustModel is supported, but YAML contains {network_type!r}.')

    model = SandDustModel(**network_options)
    checkpoint_path = resolve_path(config['test']['checkpoint'])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Checkpoint was not found: {checkpoint_path}\n'
            'Train the model first or update test.checkpoint in the YAML file.')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'params_ema' in checkpoint:
        parameters = checkpoint['params_ema']
    elif isinstance(checkpoint, dict) and 'params' in checkpoint:
        parameters = checkpoint['params']
    else:
        parameters = checkpoint
    model.load_state_dict(parameters, strict=True)
    model.to(device).eval()
    return model, checkpoint_path


def load_image(path, resize_long_side):
    image = Image.open(path).convert('RGB')
    original_size = image.size
    resized = False
    if resize_long_side > 0 and max(image.size) > resize_long_side:
        scale = resize_long_side / max(image.size)
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        resized = True

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size, resized


def save_image(tensor, path, original_size=None):
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), 'RGB')
    if original_size is not None:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description='Test sand_dust model.')
    parser.add_argument(
        '--config',
        default=str(PROJECT_ROOT / 'options' / 'sand_dust.yml'),
        help='Path to the single YAML configuration file.')
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    test_options = config['test']
    input_dir = resolve_path(test_options['input_dir'])
    result_dir = resolve_path(test_options['result_dir'])
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Test input folder was not found: {input_dir}')

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f'No test images were found in: {input_dir}')

    use_gpu = config.get('num_gpu', 1) > 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    model, checkpoint_path = load_model(config, device)
    resize_long_side = int(test_options.get('resize_long_side', 0))
    resize_back = bool(test_options.get('resize_back', True))

    print('Model: sand_dust model')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Device: {device}')
    print(f'Input images: {len(image_paths)}')

    result_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for image_path in tqdm(image_paths, unit='image'):
            input_tensor, original_size, resized = load_image(
                image_path, resize_long_side)
            restored = model(input_tensor.to(device))
            resize_to = original_size if resized and resize_back else None
            save_image(restored, result_dir / f'{image_path.stem}.png', resize_to)

    print(f'Finished. Results: {result_dir}')


if __name__ == '__main__':
    main()
