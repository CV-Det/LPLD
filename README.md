# LPLD (Low-confidence Pseudo-Label Distillation) (ECCV 2024)

This is an official code implementation repository for ```Enhancing Source-Free Domain Adaptive Object Detection with Low-Confidence Pseudo-Label Distillation```, accepted to ```ECCV 2024```.

<p align="center">
  <img src="https://github.com/junia3/LPLD/assets/79881119/1f217e54-4a3b-4be5-abdb-c924af1026f1">
</p>

---

## Installation and Environmental settings (Instructions)

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

## Dataset preparation
- Cityscapes, FoggyCityscapes / [Download Webpage](https://www.cityscapes-dataset.com/) / [Google drive (preprocessed)](https://drive.google.com/file/d/1A2ak_gjkSIRB9SMANGBGTmRoyB10TTdB/view?usp=sharing)
- PASCAL_VOC / [Download Webpage](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)
- Clipart / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) / [Google drive (preprocessed)](https://drive.google.com/file/d/1IH6zX-BBfv3XBVY5i-V-4oTLTj39Fsa6/view?usp=sharing)
- Watercolor / [Download Webpage](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) / [Google drive (preprocessed)](https://drive.google.com/file/d/1H-zIRNZx3mU4SuG30PG5KrgmKw-5v9LY/view?usp=sharing)
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

## Execution

> Currently, we only provide code and results with ResNet-50 backbone baselines.
> We are planning to add VGG-16 backbone baselines and code.

Before training, please download source models from the [google drive link](https://drive.google.com/drive/folders/1-8AbGhESrpKlg1erctbxTcwAqJ8QHDoH?usp=sharing).

### Train models

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train_main.py \ 
--config-file configs/sfda/sfda_city2foggy.yaml --model-dir ./source_model/cityscape_baseline/model_final.pth
```

### Test models

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/test_main.py --eval-only \ 
--config-file configs/sfda/sfda_city2foggy.yaml --model-dir $WEIGHT_LOCATION
```

---

## Visualize

We provide visualization code. We use our trained model to detect foggy cityscapes in the ```example image```.

<p align="center">
  <img src="https://github.com/junia3/LPLD/assets/79881119/b4b98638-1681-4f21-8181-7cdce83b095a", width="400">
  <img src="https://github.com/junia3/LPLD/assets/79881119/9441055b-1f04-49a0-a78e-117925936917", width="400">
</p>

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/visualize.py \
--config-file configs/sfda/sfda_city2foggy.yaml \
--model-dir $WEIGHT_LOCATION \
--img_path $SAMPLE_LOCATION
```

---

## Results
### Low Confidence Pseudo-Label Extraction

<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/48a4d97f-c6ca-46f3-8d22-a35af345059b", width="700">
  <br>
  <img src="https://github.com/junia3/LPLD/assets/79881119/4c78e21c-2768-4206-b059-3f2ce7738b73", width="700">
</p>

### Pretrained weights (LPLD)

|Source|Target|Download Link|
|:---:|:---:|:---:|
|Cityscapes|FoggyCityscapes|[Google drive](https://drive.google.com/file/d/1dCRI95VUjB8GDdgd1eLRX1P-uQZVI_Lg/view?usp=sharing)|
|Kitti|Cityscapes|[Google drive](https://drive.google.com/file/d/1HiDibBDigbNcL7XfghLW87r5MufArutN/view?usp=sharing)|
|Sim10k|Cityscapes|[Google drive](https://drive.google.com/file/d/1-M0hKURPslgI9XtniVzZULi_SHkRud6D/view?usp=sharing)|
|Pascal VOC|Watercolor|[Google drive](https://drive.google.com/file/d/1ShvXTtsaoxAVJrdu3EXpM-ao33jc5ChQ/view?usp=sharing)|
|Pascal VOC|Clipart|[Google drive](https://drive.google.com/file/d/1y6woJCPTlaZgPrF3nv36MIK0nn8Yy28d/view?usp=sharing)|

### Cityscapes to FoggyCityscapes
<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/4269a6a0-83ad-4d64-b8b7-576f29254bf0", width="700">
</p>

### Kitti to Cityscapes
<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/bb518dba-5c03-4d00-945e-5b4a6aad6bbd", width="700">
</p>

### Sim10k to Cityscapes
<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/c2b768e9-1bb8-4faa-8838-06ee917283cf", width="700">
</p>

### VOC to Watercolor
<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/1c338dc8-116a-46f4-b8c8-42066e4fe3e8", width="700">
  <br>
  <img src="https://github.com/junia3/LPLD/assets/79881119/d97f58c3-2490-4562-b9a1-e7ec3a667201", width="700">
</p>

### VOC to Clipart
<p align="left">
  <img src="https://github.com/junia3/LPLD/assets/79881119/03244ab9-8504-4e78-9c74-bb898cfd3aea", width="700">
  <br>
  <img src="https://github.com/junia3/LPLD/assets/79881119/f8f699d3-ca8d-40a5-87f4-1ec7665d7c38", width="700">
</p>

---

## Citation

<details>
<summary>Open</summary>
  
```bibtex
TBD
```

</details>

---

## Contact
If you have any issue with code or paper, feel free to contact [```jun_yonsei@yonsei.ac.kr```](mailto:jun_yonsei@yonsei.ac.kr).
