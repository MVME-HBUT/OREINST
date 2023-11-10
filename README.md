
## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build OREINST with:

```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd OREINST
python setup.py build develop
```

If you are using docker, a pre-built image can be pulled with:

```
docker pull tianzhi0549/adet:latest
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
sh train.sh
```
To evaluate the model after training, run:

```
sh eval.sh
```
Note that:
- The configs are made for 1-GPU training. To train on another number of GPUs, change the `--num-gpus`.
- If you want to measure the inference time, please change `--num-gpus` to 1.
- We set `OMP_NUM_THREADS=0` by default, which achieves the best speed on our machines, please change it as needed.

### Citing
If you find this repository useful in your research, please consider citing:
```
@ARTICLE{OREINST2023,  
  author={Guodong Sun and Delong Huang and Yuting Peng and Le Cheng and Bo Wu and Yang Zhang},  
  booktitle={Engineering Applications of Artificial Intelligence},   
  title={Efficient Segmentation with Texture in Ore Images Based on Box-supervised Approach},   
  year={2023},
  pages={1-14}
  }
```
