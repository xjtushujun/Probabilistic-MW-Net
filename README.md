# Probabilistic-MW-Net (PMW-Net)
TNNLS2021: A Probabilistic Formulation for Meta-Weight-Net (Official Pytorch implementation)


================================================================================================================================================================


This is the code for the paper:
A Probabilistic Formulation for Meta-Weight-Net. Qian Zhao , Jun Shu, Xiang Yuan, Ziming Liu, and Deyu Meng. [Official site](https://ieeexplore.ieee.org/abstract/document/9525050/) ,  [Copy Vervision](https://github.com/xjtushujun/Probabilistic-MW-Net/blob/main/PMWNet_journal.pdf)

[open-set noise](https://drive.google.com/file/d/1v-ZmUx4NZEvCTEYW1kVShdHAbB5zdF1c/view?usp=sharing) in our paper is constructed following [Iterative learning with open-set noisy labels](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Iterative_Learning_With_CVPR_2018_paper.pdf)


# Setups
The requiring environment is as bellow:

Linux

Python 3+

PyTorch 0.4.0

Torchvision 0.2.0

# Running Meta-Weight-Net on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example:


```
python PMWN.py --dataset cifar10 --corruption_type unif(flip2) --corruption_prob 0.6
```

```
python PMWN.py --dataset cifar10 --corruption_type open_set_cifar100(open_set_imagenet) --corruption_prob 0.6
```


If you find this code useful in your research then please cite  
```bash
@article{zhao2021probabilistic,
  title={A Probabilistic Formulation for Meta-Weight-Net},
  author={Zhao, Qian and Shu, Jun and Yuan, Xiang and Liu, Ziming and Meng, Deyu},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
``` 
