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
- Cityscapes, Foggy cityscapes / [Download Webpage](https://www.cityscapes-dataset.com/) / [Direct Download (preprocessed)]()
- PASCAL_VOC / [Download Webpage](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) / [Direct Download (preprocessed)]()
- Watercolor, Clipart / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) / [Direct Download (preprocessed)]()

Make sure that all downloaded datasets are located in the ```./dataset``` folder. All dataset codes are written to fit the format of PASCAL_VOC.
After preparing the datasets, you will have the following file structure:

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
