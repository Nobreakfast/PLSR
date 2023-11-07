# PLSR: Unstructured Pruning with Layer-wise Sparsity Ratio

This project aimed to cover the experiments of [our paper]() in ICMLA2023. If your like this work, please cite our paper:

```
@article{zhao2023pslr,
  title={PSLR: Unstructured Pruning with Layer-wise Sparsity Ratio},
  author={Zhao, Haocheng and Yu, Limin and Guan, Runwei and Jia, Liye and Zhang, Junqing and Yue, Yutao},
  journal={22nd IEEE Internaltional Conference on Machine Learning and Application},
  year={2023}
}
```

## Requirements

```shell
# pytorch
pip install torch==1.11.0 torchvision==0.12.0

# others
pip install hydra-core==1.2 tqdm tensorboardX
```

## Paper Experiments

### Pre-trained models

``` shell
sh exp/bl.sh
```

### Figure

``` shell
# pre-trained models
sh exp/exp0_bl.sh

# figure
sh exp/exp0_trained.sh
```

### Table

``` shell
# Table II
sh exp/exp1.sh

# Table III
sh exp/exp2.sh

# Table IV
sh exp/exp3_erk.sh
sh exp/exp3_featio.sh

# Table V
sh exp/exp_ablation.sh
sh exp/exp_ablation2.sh
```
