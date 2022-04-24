# I-BAU: Adversarial-Unlearning-of-Backdoors-via-Implicit-Hypergradient
Official Implementation of ICLR 2022 paper, **Adversarial Unlearning of Backdoors via Implicit Hypergradient** \[[arxiv](https://arxiv.org/pdf/2110.03735.pdf)\]\[[openreview](https://openreview.net/forum?id=MeeQkFYVbzW)\]
We propose a novel minimax formulation for removing backdoors from a given poisoned model based on a small set of clean data:
<img src="https://latex.codecogs.com/svg.image?\theta^{*}=\underset{\theta}{\arg&space;\min&space;}&space;\max&space;_{\|\delta\|&space;\leq&space;C_{\delta}}&space;H(\delta,&space;\theta):=\frac{1}{n}&space;\sum_{i=1}^{n}&space;L\left(f_{\theta}\left(x_{i}&plus;\delta\right),&space;y_{i}\right)&space;">
<br>

 To solve the minimax problem, we propose **the Implicit Backdoor Adversarial Unlearning (I-BAU) algorithm**, which utilizes the implicit hypergradient to account for the interdependence between inner and outer optimization. I-BAU requires less computation to take effect; particularly, it is more than <span style="color:blue">some **13X faster** text</span> than the most efficient baseline in the single-target attack setting. It can still remain effective in the extreme case where the defender can <span style="color:blue">some **only access 100 clean samples** text</span> — a setting where <span style="color:blue">some **all the baselines fail to produce acceptable results** text</span>.

## Requirement
This code has been tested with Python 3.6, PyTorch 1.8.1 and cuda 10.1. 

## Getting Started
* Install required packages.
* Get poisoned models prepared in the directory `./checkpoint/`. <br>
* We provide two unlearn examples of I-BAU on poisoned models trained on GTSRB and CIFAR10 datasets, check <br> `clean_solution_batch_opt_hyperdimentional_cifar.ipynb` <br> and <br> `clean_solution_batch_opt_hyperdimentional_gtsrb.ipynb` for more details.
* For a more flexible usage, run `python defense.py`. An example is as follow:<br>
`python defense.py --dataset cifar10 --poi_path './checkpoint/badnets_8_02_ckpt.pth'  --optim Adam --lr 0.001 --n_rounds 3 --K 5`. <br>
Clean data used for backdoor unlearning can be specified with argument `--unl_set`; if it is not specified, then a subset of data from testset will be used for unlearning. <br>
* For more information regarding training options, please check the help message: <br>
`python defense.py --help`. <br>

## Poster
<center><img src="http://www.yi-zeng.com/wp-content/uploads/2022/04/ICLR-Poster.png"></center>

## Citation
If you find our work useful please cite:
```
@article{zeng2021adversarial,
  title={Adversarial Unlearning of Backdoors via Implicit Hypergradient},
  author={Zeng, Yi and Chen, Si and Park, Won and Mao, Z Morley and Jin, Ming and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2110.03735},
  year={2021}
}
```
