# DexArt Bucket Language Planning Labeling

We use the Bucket task in the DexArt dataset as an example to illustrate the labeling process. This example includes using a pure visual pipeline to annotate the bounding boxes of objects and leveraging camera intrinsic and extrinsic parameters to obtain the gripper position.

It is possible to estimate gripper positions using a pure visual pipeline—such as the method proposed in [Embodied-CoT](https://github.com/MichalZawalski/embodied-CoT). However, we find that utilizing accurate camera parameters yields better results. Although the DexArt dataset is based on simulation, directly obtaining usable segmentation masks from the simulator is non-trivial. Therefore, we adopt a vision foundation model to assist in the annotation. This pipeline is also applicable to real-world data.

## Usage

Ensure you have installed git lfs:
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### Dataset Download

VLA-OS-Dataset provides commands to download all the datasets used in VLA-OS paper. Here you can only download DexArt Bucket dataset as follows.
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/Linslab/VLA-OS-Dataset
git lfs pull -I dexart/bucket_dex_art_dataset
```

Then you can move the dataset to `data/tfds_datasets` (Create `data/tfds_datasets` first if you don't have).
```bash
mv VLA-OS-Dataset/dexart/bucket_dex_art_dataset VLA-OS-Language-Planning-Labeling/data/tfds_datasets
cd VLA-OS-Language-Planning-Labeling
```

### Grounded-SAM-2 Installation

Pull the submodule in `third_party/Grounded-SAM-2`.
```bash
git submodule update --init --recursive
```

Next, you can follow the instructions in `third_party/Grounded-SAM-2` for installation.

### Labeling with Grounded-SAM-2 

Finally, you can follow the instructions in `scripts/dexart_scripts/dexart_pipeline.ipynb` to implement language planning labeling.