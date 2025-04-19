# Large coordinate attention network for lightweight image super-resolution
# LCAN
Fangwei Hao, Jiesheng Wu, Haotian Lu, Ji Du, Jing Xu, Xiaoxuan Xu,

paper link: https://arxiv.org/abs/2405.09353
 ## Abstract
The Multi-Scale Receptive Field (MSRF) and Large Kernel Attention (LKA) module have been
shown to significantly improve performance in the lightweight image super-resolution (SR) task.
However, existing lightweight SR methods seldom pay attention to designing lightweight yet effective
building block with MSRF for local modeling, and their LKA modules face a quadratic increase
in computational and memory footprints as the convolutional kernel size increases. To address the
first issue, we propose a simple but effective block, Multi-scale Blueprint Separable Convolutions
(MBSConv), as highly efficient building block with MSRF, and it can focus on the learning for the
multi-scale information which is a vital component of discriminative representation. As for the second
issue, in order to mitigate the complexity of LKA, we propose a Large Coordinate Kernel Attention
(LCKA) module which decomposes the two-dimensional convolutional kernels of the depth-wise
convolutional layers in LKA into horizontal and vertical one-dimensional kernels. LCKA enables
the adjacent direct interaction of local information and long-distance dependencies not only in the
horizontal direction but also in the vertical. Besides, LCKA allows for the direct use of extremely
large kernels in the depth-wise convolutional layers to capture more contextual information which
helps to significantly improve the reconstruction performance, while incurring lower computational
complexity and memory footprints. Integrating MBSConv and LCKA, we propose a Large Coordinate
Attention Network (LCAN) which is an extremely lightweight SR network with efficient learning
capability for local, multi-scale, and contextual information. Extensive experiments show that our
LCAN with extremely low model complexity achieves superior performance compared to previous
lightweight state-of-the-art SR methods.

## Environment
PyTorch >= 1.11 (Recommend >= 1.11)
BasicSR = 1.4.2

## Installation
For installing, follow these instructions:
~~~
pip install -r requirements.txt
python setup.py develop
~~~

## Citation
If you find this project useful for your research, please consider citing:
@article{hao2024large,
  title={Large coordinate kernel attention network for lightweight image super-resolution},
  author={Hao, Fangwei and Wu, Jiesheng and Lu, Haotian and Du, Ji and Xu, Jing and Xu, Xiaoxuan},
  journal={arXiv preprint arXiv:2405.09353},
  year={2024}
}
## Acknowledgements
This code is built on LKDN [https://github.com/stella-von/LKDN]. We thank the authors for sharing their codes of  PyTorch version.
## Contact
Should you have any question, please contact Fangwei Hao.
