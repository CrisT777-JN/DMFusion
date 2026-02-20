# DMFusion (IEEE T-CSVT2026)
DMFusion: Degradation-Customized Mixture-of-Experts with Adaptive Discrimination for Multi-Modal Image Fusion
# Setup

```bash
# create and activate the environment
conda create -n DMFusion python=3.8
conda activate DMFusion

# install required libraries
pip install numpy==1.24.4
pip install torch==2.4.1
pip install torchvision==0.19.1
pip install tqdm==4.67.1
pip install timm==1.0.19
pip install einops==0.8.1
pip install git+https://github.com/openai/CLIP.git
```

# Dataset & Pre-trained weights

The test set is located in the `./test_data/` directory.

If you want to test on your own dataset, please arrange the files in the structure below:

```text
your_dataset/
├── Inf/   # infrared images
└── Vis/   # visible images (RGB)
```
- **Full Datasets** | Original datasets used in our experiments | [Google Drive](https://drive.google.com/drive/folders/1G0YlkSEuk6NXodZM1NKuiVCS7u3fn7tq?usp=sharing)

Place the pretrained weights as follows:

```text
runs/
├── best_cls.pth   # CLIP
├── F_Base.pth     # restoration branch weights
└── F.pth          # fusion network weights
```
We provide the pre-trained models (checkpoints) for datasets to facilitate reproduction.

- **Google Drive**: [Download Link](https://drive.google.com/drive/folders/1XkZGUvEBA0DffiiHUTuEz_hTMOkMLWjF?usp=sharing)

# Training

The training process consists of two stages:

**Stage 1:**
```bash
python train_main0.py
```
**Stage 2:**
```bash
python train_main1.py
```

# Testing

If you want to infer with our DMFusion and obtain the fusion results in our paper, please run `test_all.py`.

Then, the fused results will be saved in the `./test_data/output_fx/` folder, the repaired infrared images will be saved in the `./test_data/output_ir/` folder, and the repaired visible light images will be saved in the `./test_data/output_vi/` folder.

If you just want to quickly test our fusion results, you can choose to run `test_params_fps.py` for a faster test.

# Citation

If you find our work useful in your research, please consider citing:

```bibtex
@ARTICLE{Chen2026DMFusion,
  author={Chen, Tao and Wang, Chuang and Zhang, Yudong and Xia, Kaijian and Qian, Pengjiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DMFusion: Degradation-Customized Mixture-of-Experts with Adaptive Discrimination for Multi-Modal Image Fusion}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
}
```
