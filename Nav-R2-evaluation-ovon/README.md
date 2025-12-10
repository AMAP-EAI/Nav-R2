
# NaVid 

**Video-Language in, Actions out! No odometry, Dpeth, or Map required!** This project contains the evaluation code of our RSS 2024 paper:

 **NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**.

Contributors: [Jiazhao Zhang](https://jzhzhang.github.io/), Kunyu Wang, [Rongtao Xu](https://scholar.google.com.hk/citations?user=_IUq7ooAAAAJ), [Gengze Zhou](https://gengzezhou.github.io/), [Yicong Hong](https://yiconghong.me/), Xiaomeng Fang, [Qi Wu](http://qi-wu.me/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/)<br>

[[Paper & Appendices](https://arxiv.org/pdf/2402.15852)] [[Projece Page](https://pku-epic.github.io/NaVid/)]



https://github.com/user-attachments/assets/eb545ef0-516c-4b6a-92a8-225f839843cc



## Prerequisites 

### 1. Installation

To begin, you'll need to set up the [Habitat VLN-CE](https://github.com/jacobkrantz/VLN-CE) environment. The first step is to install [Habitat-sim (version 0.1.7)](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) with Python 3.8. We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) or [Anaconda](https://www.anaconda.com/) for easy installation:


```
conda create -n vlnce_navid python=3.8
conda activate vlnce_navid
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
```

**Another option is to download the habitat-sim package from the [conda website](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2) then use the command `conda install name_of_package.tar.bz2` to install the habitat-sim in your conda environment. This method is more stable as it avoids potential network issues or problems associated with direct conda installation.**


Next, install [Haibtat-Lab 0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7). 
```
mkdir navid_ws | cd navid_ws
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git 
or
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```
Finally, install NaVid:
```
cd ..
git clone git@github.com:jzhzhang/NaVid-VLN-CE.git
or 
git clone https://github.com/jzhzhang/NaVid-VLN-CE.git

cd NaVid-VLN-CE
pip install -r requirements.txt
```

### 2. Vision-and-Language Data

Follow the instructions in the [VLN-CE Data Section](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#data) to set up the scene dataset and episodes dataset. (If the RxR episodes cannot be accessed, you can download them [here](https://1drv.ms/u/c/aa19f644cf9d8afb/ETQ8Co-hGLFMjwd5HckKsvABjWvZ3cbPsWwdzbhmQDoL1g?e=WtO8Lm).) After completing the data preparation, update the data location in [R2R config file](VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml) and [RxR config file](VLN_CE/habitat_extensions/config/vlnce_task_navid_rxr.yaml). An example configuration is shown below, please modify the task files to align your data configuration:
```
NDTW:
  GT_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz 
DATASET:
  TYPE: VLN-CE-v1 # for R2R 
  SPLIT: val_unseen
  DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz # episodes dataset
  SCENES_DIR: data/scene_datasets/ # scene datasets 

```

### 3. Pretrained Weights
First download the pretriend weights for vision encoder [EVA-ViT-G](https://github.com/dvlab-research/LLaMA-VID/tree/main). Then, download the [finetuned NaVid model](https://huggingface.co/Jzzhang/NaVid/tree/main). The model has been trained on extensive samples from the `training splits` of the VLN-CE R2R and RxR datasets, following the training strategy of [Uni-NaVid](https://arxiv.org/pdf/2412.06224).


| Evaluation Benchmark |  TL  |  NE  |  OS  |  SR  |  SPL |
|----------------------|:----:|:----:|:----:|:----:|:----:|
| VLN-CE R2R Val.      | 10.7 | 5.65 | 49.2 | 41.9 | 36.5 |
| [VLN-CE R2R Test](https://eval.ai/web/challenges/challenge-page/719/leaderboard/1966)      | 11.3 | 5.39 |  52  |  45  |  39  |
| VLN-CE RxR Val.      | 15.4 | 5.72 | 55.6 | 45.7 | 38.2 |


### 4.  Structure
We recommend organizing your project directory as follows
```
navid_ws
├── habitat-lab
├── NaVid-VLN-CE
│   ├── navid
│   ├── VLN_CE
│   ├── model_zoo
│   │   ├── eva_vit_g.pth
│   │   ├── <navid_weights>
```


## Evaluation

To evaluate the model on multiple GPUs, use the provided evaluation `eval_navid_vlnce.sh` script. Each GPU will handle one split of all episodes. Before running the script, modify the environment variables as follows:

```
CHUNKS=8 # GPU numbers
MODEL_PATH="" # model wieght
CONFIG_PATH="" # task configuration configure, see script for an example
SAVE_PATH="" #  results
```

Our evaluation code supports the reuse of historical visual tokens, enabling the completion of 1839 episodes (R2R) within 2.5 hours (using 8 A100 GPUs). Run the script with:
```
bash eval_navid_vlnce.sh
```

Results will be saved in the specified `SAVE_PATH`, which will include a `log` directory and a `video` directory. To monitor the results during the evaluation process, run:

```
watch -n 1 python  analyze_results.py --path YOUR_RESULTS_PATH
```
To stop the evaluation, use:
```
bash kill_navid_eval.sh
```


## Citation
If you find this work useful for your research, please consider citing:
```
@article{zhang2024navid,
        title={NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation},
        author={Zhang, Jiazhao and Wang, Kunyu and Xu, Rongtao and Zhou, Gengze and Hong, Yicong and Fang, Xiaomeng and Wu, Qi and Zhang, Zhizheng and Wang, He},
        journal={Robotics: Science and Systems},
        year={2024}
      }
```

## Acknowledgments
Our code is based on [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE). 

This is an open-source version of NaVid, some functions have been rewritten to avoid certain license. 

If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.
