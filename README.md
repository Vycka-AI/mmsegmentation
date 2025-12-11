# Semantic Segmentation (fork fromn MMSegmentation)

## 1. Setup

First, clone the repository:

```bash
git clone https://github.com/Vycka-AI/mmsegmentation
```

Then, prepare the environment using Conda.


**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mm_env python=3.10 -y
conda activate mm_env
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices to install MMSegmentation. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

```shell
cd mmsegmentation
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

```shell
pip install "mmsegmentation>=1.0.0"
```

## 2. Training

###Getting Data
```shell
cd mmsegmentation
mkdir data && cd data
#Copy your data here and unzip it
cp /eirt_output.zip . #Just an example
unzip eirt_output.zip
```

###**Creating train test splits**

Copy "create_splits.py" file to eirt_output folder
```shell
#First create splits
cd mmsegmentation
cp create_splits.py data/eirt_output/batch01/
cd data/eirt_output/batch01/
python create_splits.py
```

**Run training**
To run training:
```shell
python tools/train.py configs/3_Class.py --work-dir work_dirs/3_Class_Debug
```

## 3. Inference
To test out images, currently it reads some of the images from "'data/real/val.txt'". Change it accordingly to your work directory.

```shell
python inference_3Class.py
```
