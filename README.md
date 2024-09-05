# tqdm
This repository is the official pytorch implementation of **tqdm** introduced by:
### [**Textual Query-Driven Mask Transformer for Domain Generalized Segmentation (ECCV 2024)**](https://arxiv.org/abs/2407.09033)

## Environment
### Requirements
- The requirements can be installed with:
  
  ```
  pip install -r requirements.txt
  ```
### Pre-trained VLM Models
- Please download the pre-trained CLIP and EVA02-CLIP and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | CLIP | `ViT-B-16.pt` |[official repo](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B` |[official repo](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#eva-02-clip-series)|

### Checkpoints
- You can download **tqdm** model checkpoints.

  | model | link |
  |-----|:-----:|
  | `tqdm-clip-vit-b-gta` |[download link](https://drive.google.com/drive/u/1/folders/1x99C0EZY_SGwt1dbiur54wbR6iybiF9s)|
  | `tqdm-eva02-clip-vit-l-gta` |[download link](https://drive.google.com/drive/u/1/folders/1x99C0EZY_SGwt1dbiur54wbR6iybiF9s)|
  | `tqdm-eva02-clip-vit-l-city` |[download link](https://drive.google.com/drive/u/1/folders/1x99C0EZY_SGwt1dbiur54wbR6iybiF9s)|

## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files]() following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR DATA FOLDER ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR DATA FOLDER ROOT]', ...)
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
- [configs/tqdm/*]() - Config files for the final tqdm
- [models/segmentors/*]() - Overall tqdm framework
- [mmseg/models/utils/assigner.py]() - Implementation of fixed matching
- [mmseg/models/decode_heads/tqdm_head.py]() - Our textual object query-based segmentation head
- [mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py]() - Our pixel decoder with *text-to-pixel attention*

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DAFormer](https://github.com/lhoyer/DAFormer)
- [TLDR](https://github.com/ssssshwan/TLDR)
