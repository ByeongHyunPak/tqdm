# [ECCV 2024] Textual Query-Driven Mask Transformer for Domain Generalized Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/textual-query-driven-mask-transformer-for/domain-generalization-on-gta5-to-cityscapes)](https://paperswithcode.com/sota/domain-generalization-on-gta5-to-cityscapes?p=textual-query-driven-mask-transformer-for) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/textual-query-driven-mask-transformer-for/domain-generalization-on-gta-to-avg)](https://paperswithcode.com/sota/domain-generalization-on-gta-to-avg?p=textual-query-driven-mask-transformer-for) <br />
### [**Textual Query-Driven Mask Transformer for Domain Generalized Segmentation**](https://arxiv.org/abs/2407.09033)
>[Byeonghyun Pak](https://byeonghyunpak.github.io/)\*, [Byeongju Woo](https://byeongjuwoo.github.io/)\*, [Sunghwan Kim](https://sunghwan.me/)\*, [Dae-hwan Kim](https://scholar.google.com/citations?hl=en&user=_5Scn8YAAAAJ), [Hoseong Kim](https://scholar.google.com/citations?hl=en&user=Zy7Sz5UAAAAJ)†\
>Agency for Defense Development\
>ECCV 2024

#### [[`Project Page`](https://byeonghyunpak.github.io/tqdm/)] [[`Paper`](https://arxiv.org/abs/2407.09033)]

## Environment
### Requirements
- The requirements can be installed with:
  
  ```bash
  conda create -n tqdm python=3.9 numpy=1.26.4
  conda activate tqdm
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install xformers==0.0.20
  pip install mmcv-full==1.5.3 
  ```
### Pre-trained VLM Models
- Please download the pre-trained CLIP and EVA02-CLIP and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | CLIP | `ViT-B-16.pt` |[official repo](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B` |[official repo](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#eva-02-clip-series)|

### Checkpoints
- You can download **tqdm** model checkpoints:

  | Model | Config | Link |
  |-----|-----|:-----:|
  | `tqdm-clip-vit-b-gta` | [config](https://github.com/ByeongHyunPak/tqdm/blob/main/configs/tqdm/tqdm_clip_vit-l_1e-5_20k-g2c-512.py) |[download link](https://drive.google.com/file/d/1PAnjxDUQ1nrUkDne1jflz_2BJQgtoqmZ/view?usp=drive_link)|
  | `tqdm-eva02-clip-vit-l-gta` | [config](https://github.com/ByeongHyunPak/tqdm/blob/main/configs/tqdm/tqdm_eva_vit-l_1e-5_20k-g2c-512.py) |[download link](https://drive.google.com/file/d/1GHR0R5A06oPlRsM1XRu1zmFUJa2_WOSi/view?usp=drive_link)|
  | `tqdm-eva02-clip-vit-l-city` | [config](https://github.com/ByeongHyunPak/tqdm/blob/main/configs/tqdm/tqdm_eva_vit-l_1e-4_20k-c2b-512.py) |[download link](https://drive.google.com/file/d/13taMQNPWIP4yrzUsWvXBG7LMY5lsbOp-/view?usp=drive_link)|

## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files](https://github.com/ByeongHyunPak/tqdm/tree/main/configs/_base_/datasets) following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```
## Train
 ```
 bash dist_train.sh configs/[TRAIN_CONFIG] [NUM_GPUs]
```
  - `[TRAIN_CONFIG]`: train configuration (e.g., `tqdm/tqdm_eve_vit-l_1e-5_20k-g2c-512.py`)
  - `[NUM_GPUs]`: the number of the GPUs
## Test
```
bash dist_test.sh configs/[TEST_CONFIG] work_dirs/[MODEL] [NUM_GPUs] --eval mIoU
```
  - `[TRAIN_CONFIG]`: test configuration (e.g., `tqdm/tqdm_eve_vit-l_1e-5_20k-g2b-512.py`)
  - `[MODEL]`: model checkpoint (e.g., `tqdm_eve_vit-l_1e-5_20k-g2c-512/epoch_last.pth`)
  - `[NUM_GPUs]`: the number of the GPUs
 
## The Most Relevant Files
- [configs/tqdm/*](https://github.com/ByeongHyunPak/tqdm/tree/main/configs/tqdm) - Config files for the final tqdm
- [models/segmentors/*](https://github.com/ByeongHyunPak/tqdm/tree/main/models/segmentors) - Overall tqdm framework
- [mmseg/models/utils/assigner.py](https://github.com/ByeongHyunPak/tqdm/blob/main/mmseg/models/utils/assigner.py#L168) - Implementation of fixed matching
- [mmseg/models/decode_heads/tqdm_head.py](https://github.com/ByeongHyunPak/tqdm/blob/main/mmseg/models/decode_heads/tqdm_head.py) - Our textual object query-based segmentation head
- [mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py](https://github.com/ByeongHyunPak/tqdm/blob/main/mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py) - Our pixel decoder with *text-to-pixel attention*

## Citation
If you find our code helpful, please cite our paper:
```bibtex
@inproceedings{pak2024textual,
  title={Textual Query-Driven Mask Transformer for Domain Generalized Segmentation},
  author={Pak, Byeonghyun and Woo, Byeongju and Kim, Sunghwan and Kim, Dae-hwan and Kim, Hoseong},
  booktitle={European conference on computer vision},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DAFormer](https://github.com/lhoyer/DAFormer)
- [TLDR](https://github.com/ssssshwan/TLDR)
