# LPLD (Low-confidence Pseudo-Label Distillation) (ECCV 2024)

Official repository for ```Enhancing Source-Free Domain Adaptive Object Detection with Low-Confidence Pseudo-Label Distillation```, accepted to ```ECCV 2024```.

<p align="center">
  <img src="https://github.com/junia3/LPLD/assets/79881119/1f217e54-4a3b-4be5-abdb-c924af1026f1">
</p>

---

### Installation and Environmental settings (Instructions)

- We use Python 3.6 and Pytorch 1.9.0
- The codebase from [Detectron2](https://github.com/facebookresearch/detectron2).

```bash
git clone https://github.com/junia3/LPLD.git

conda create -n LPLD python=3.6
conda activate LPLD
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

cd LPLD
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e LPLD

```

---

### Dataset preparation
- Cityscapes / [Download Webpage](https://www.cityscapes-dataset.com/) / [Direct Download (preprocessed)](https://drive.google.com/file/d/1A2ak_gjkSIRB9SMANGBGTmRoyB10TTdB/view?usp=sharing)
- Foggy cityscapes / [Download Webpage](https://www.cityscapes-dataset.com/) / [Direct Download (preprocessed)]()
- PASCAL_VOC / [Download Webpage](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
- Clipart / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) / [Direct Download (preprocessed)](https://drive.google.com/file/d/1IH6zX-BBfv3XBVY5i-V-4oTLTj39Fsa6/view?usp=sharing)
- Watercolor / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) / [Direct Download (preprocessed)]()
- Sim10k / [Download Webpage](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)

Make sure that all downloaded datasets are located in the ```./dataset``` folder. After preparing the datasets, you will have the following file structure:

```bash
LPLD
...
├── dataset
│   └── foggy
│   └── cityscape
│   └── clipart
│   └── watercolor
...
```
Make sure that all dataset fit the format of PASCAL_VOC. For example, the dataset foggy is stored as follows:

```bash
$ cd ./dataset/foggy/VOC2007/
$ ls
Annotations  ImageSets  JPEGImages
$ cat ImageSets/Main/test_t.txt
target_munster_000157_000019_leftImg8bit_foggy_beta_0.02
target_munster_000124_000019_leftImg8bit_foggy_beta_0.02
target_munster_000110_000019_leftImg8bit_foggy_beta_0.02
.
.
```

---

### Pretrained weights

- Source Model
  - Cityscapes / [Download Link]()
  - Sim10k / [Download Link]()
  - Kitti / [Download Link]()
  - PASCAL VOC / [Download Link]()
    
- Ours
  - Cityscapes to FoggyCityscapes / [Download Link]()
  - Sim10k to Cityscapes / [Download Link]()
  - Kitti to Cityscapes / [Download Link]()
  - VOC to Clipart / [Download Link]()
  - VOC to watercolor / [Download Link]()
