# 

# Nav-R2
<!-- [![paper](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/) -->
Official Implementation of paper: ```Nav-R2:Dual‑Relation Reasoning for Generalizable Open‑Vocabulary Object‑Goal Navigation```



## Abstract
Object-goal navigation in open-vocabulary settings requires agents to locate novel objects in unseen environments, yet existing approaches suffer from opaque decision-making processes and low success rate on locating unseen objects.
To address these challenges, we propose Nav-R2, a framework that explicitly models two critical types of relationships, target-environment modeling and environment-action planning, through structured Chain-of-Thought (CoT) reasoning coupled with a Similarity-Aware Memory.
We construct a Nav$R^2$-CoT dataset that teaches the model to perceive the environment, focus on target-related objects in the surrounding context and finally make future action plans.
Our SA-Mem preserves the most target-relevant and current observation-relevant features from both temporal and semantic perspectives by compressing video frames and fusing historical observations, while introducing no additional parameters.
Compared to previous methods, Nav-R2 achieves state-of-the-art performance in localizing unseen objects through a streamlined and efficient pipeline, avoiding overfitting to seen object categories while maintaining real-time inference at 2Hz.

## Contributions
(1) A relational reasoning framework for object-goal navigation that explicitly models the **target-environment** (perception) and **environment–action**(planning) relationships, integrating this structured reasoning in a streamlined pipeline without introducing additional model parameters.

(2) A novel Chain-of-Thought dataset specifically designed for training a generalizable object-goal navigation model capable of reasoning and modeling both two relationships.

(3) A vision-language reasoning model, Nav-R2, just trained via supervised fine-tuning on first-person RGB frames, achieving state-of-the-art performance in open-vocabulary ObjectNav and real-time inference at around 2Hz.

## Getting started with Nav-R2

- **Training**: Install conda environment following steps below:

```shell
conda create -n Nav-R2-training
pip install -r requirements-for-training.txt
# attention:
# three libraries should be installed from source files in the environment-modules-customed folder:
# transformers, trl, and flash_attn
```

- **Evaluation**: Install conda environment following steps below:

```shell
conda create -n Nav-R2-evaluation python=3.9.19
pip install -r requirements-for-evaluation-on-OVON.txt
# attention:
# four libraries should be installed from source files in the environment-modules-customed folder:
# flash_attn, transformers, habitat_lab, and habitat-baseline
pip install environment-modules-customed/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# and please install transformers first, then habitat_lab, finally habitat-baseline
pip install -e environment-modules-customed/transformers_4.51.3-xwt-customed/transformers
pip install -e environment-modules-customed/habitat-related/habitat-lab
pip install -e environment-modules-customed/habitat-related/habitat-baselines
```

