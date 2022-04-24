# I-BAU: Adversarial-Unlearning-of-Backdoors-via-Implicit-Hypergradient
![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-DodgerBlue.svg?style=plastic)
![CUDA 10.1](https://img.shields.io/badge/cuda-10.1-DodgerBlue.svg?style=plastic)

Official Implementation of ICLR 2022 paper, **Adversarial Unlearning of Backdoors via Implicit Hypergradient** \[[openreview](https://openreview.net/forum?id=MeeQkFYVbzW)\]\[[video](https://www.youtube.com/watch?v=j8BwMYJtPdg&t=2s)\] . <br>

We propose a novel minimax formulation for removing backdoors from a given poisoned model based on a small set of clean data: <br>
<img src="https://latex.codecogs.com/svg.image?\theta^{*}=\underset{\theta}{\arg&space;\min&space;}&space;\max&space;_{\|\delta\|&space;\leq&space;C_{\delta}}&space;H(\delta,&space;\theta):=\frac{1}{n}&space;\sum_{i=1}^{n}&space;L\left(f_{\theta}\left(x_{i}&plus;\delta\right),&space;y_{i}\right)&space;"> <br>

To solve the minimax problem, we propose **the Implicit Backdoor Adversarial Unlearning (I-BAU) algorithm**, which utilizes the implicit hypergradient to account for the interdependence between inner and outer optimization. I-BAU requires less computation to take effect; particularly, it is more than <span style="color:blue"> **13 X faster** </span> than the most efficient baseline in the single-target attack setting. It can still remain effective in the extreme case where the defender can <span style="color:blue">some **only access 100 clean samples** </span> — a setting where <span style="color:blue"> **all the baselines fail to produce acceptable results** </span>.
![Picture1](https://user-images.githubusercontent.com/64983135/164996598-10da3582-791a-4ad5-8471-f7a45c12be19.png)

## Requirements
This code has been tested with Python 3.6, PyTorch 1.8.1 and cuda 10.1. 

## Usage & HOW-TO
* Install required packages.
* Get poisoned models prepared in the directory `./checkpoint/`. <br>
* We provide two examples on poisoned models trained on GTSRB and CIFAR10 datasets, check `clean_solution_batch_op..._cifar.ipynb` and `clean_solution_batch_op..._gtsrb.ipynb` for more details.
* For a more flexible usage, run `python defense.py`. An example is as follow:
```ruby
python defense.py --dataset cifar10 --poi_path './checkpoint/badnets_8_02_ckpt.pth'  --optim Adam --lr 0.001 --n_rounds 3 --K 5
```
Clean data used for backdoor unlearning can be specified with argument `--unl_set`; if it is not specified, then a subset of data from testset will be used for unlearning. <br>
* For more information regarding training options, please check the help message: <br>
`python defense.py --help`. <br>

## Poster
<center><img src="http://www.yi-zeng.com/wp-content/uploads/2022/04/ICLR-Poster.png"></center>

## Citation
If you find our work useful please cite:
```
@inproceedings{zeng2021adversarial,
  title={Adversarial Unlearning of Backdoors via Implicit Hypergradient},
  author={Zeng, Yi and Chen, Si and Park, Won and Mao, Zhuoqing and Jin, Ming and Jia, Ruoxi},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

# Special thanks to...
[![Stargazers repo roster for @YiZeng623/I-BAU](https://reporoster.com/stars/YiZeng623/I-BAU)](https://github.com/YiZeng623/I-BAU/stargazers)
[![Forkers repo roster for @YiZeng623/I-BAU](https://reporoster.com/forks/YiZeng623/I-BAU)](https://github.com/YiZeng623/I-BAU/network/members)
