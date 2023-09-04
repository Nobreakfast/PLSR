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

- Pre-trained models: ```sh exp/bl.sh```
- Experiments for Table II: ```sh exp/exp_table_2.sh```
- Experiments for Table III: ```sh exp/exp_table_3.sh```
- Ablation Experiments for Table IV: ```sh exp/exp_table_4.sh```    
- Ablation Experiments for Table V: ```sh exp/exp_table_5.sh```
- Experiments for Figure 1: ```sh exp/exp_fig_1.sh```
- Discussion Experiments for Figure 3: Open the jupyter file `exp/plot.ipynb`.