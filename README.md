# Sand Dust Image Restoration

This repository provides training, inference, and evaluation code for an
underwater sand-dust image-restoration network.

The project uses one configuration file:

```text
options/sand_dust.yml
```

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU is strongly recommended
- A CUDA-compatible PyTorch installation

Install PyTorch and torchvision for your CUDA version first. Then install the
remaining packages:

```bash
pip install -r requirements.txt
```

## Project layout

```text
ISPL_SandDust_termproject/
|-- train.py
|-- test.py
|-- compute_metrics.py
|-- options/
|   `-- sand_dust.yml
|-- basicsr/
|-- datasets/
|   |-- train/
|   |   |-- input/
|   |   `-- target/
|   `-- test/
|       |-- input/
|       `-- target/
|-- experiments/
`-- results/
```

## Dataset

Download the training and test datasets from
[SandDust_Data](https://drive.google.com/drive/folders/1KA9RoKzZY-JJQFwPijljhzne-4sDXbco?usp=drive_link).

After downloading and extracting the dataset, arrange the files as follows:

```text
datasets/
|-- train/
|   |-- input/
|   `-- target/
`-- test/
    |-- input/
    `-- target/
```

The `input` directories contain degraded images, and the `target` directories
contain their clean ground-truth images.

## Training data

Place degraded sand-dust images in:

```text
datasets/train/input
```

Place their clean ground-truth images in:

```text
datasets/train/target
```

The default file-name convention supports multiple degraded colors for each
ground-truth image:

```text
datasets/train/target/0001.jpg
datasets/train/input/0001_#C89463.JPG
datasets/train/input/0001_#C6853D.JPG
```

The available color suffixes and file extensions are configured in
`options/sand_dust.yml`.

### Validation split

Training and validation use the same data folders. Ground-truth images are
split automatically so that one image cannot appear in both sets.

```yaml
val_ratio: 0.1
split_seed: 1143
```

The default validation ratio is 10%. Keep the same `split_seed` in the train
and validation sections to obtain a deterministic, non-overlapping split.

## Train

Start training from the project root:

```bash
python train.py
```

Training outputs are written to:

```text
experiments/sand_dust_model
```

Important checkpoints include:

```text
experiments/sand_dust_model/models/net_g_best.pth
experiments/sand_dust_model/models/net_g_latest.pth
```

`net_g_best.pth` is updated when validation PSNR improves. If a saved training
state exists, training resumes automatically from the most recent state.

## Restore test images

Place images to restore in:

```text
datasets/test/input
```

The default configuration loads the following checkpoint:

```text
experiments/sand_dust_model/models/net_g_best.pth
```

Run training first to create this checkpoint, or place a compatible trained
checkpoint at that path.

Run:

```bash
python test.py
```

Restored images are saved to:

```text
results/sand_dust_model
```

The test script accepts PNG, JPG, JPEG, BMP, and TIFF files. Images are resized
to a maximum long side of 512 pixels during inference to limit GPU memory use.
When `resize_back: true`, saved results are resized back to the original image
dimensions.

Ground-truth images are not required to run `test.py`.

## Calculate PSNR and SSIM

To evaluate restored images, place matching ground-truth images in:

```text
datasets/test/target
```

The result and ground-truth file stems must match. Extensions may differ:

```text
results/sand_dust_model/0001.png
datasets/test/target/0001.jpg
```

Run:

```bash
python compute_metrics.py
```

The script prints per-image and average PSNR/SSIM values. Evaluation uses the Y
channel by default. Set `test.test_y_channel: false` in the YAML file to use
RGB evaluation.

## Configuration

All paths and training parameters are stored in `options/sand_dust.yml`.
Relative paths are resolved from the project directory, so the repository can
be moved without modifying the source code.
