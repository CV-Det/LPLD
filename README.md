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
