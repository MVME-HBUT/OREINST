
# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

For demo, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/Aabn3BEuq4HKiNK/download -O BoxInst_MS_R_50_3x.pth
python demo/demo.py \
    --config-file configs/BoxInst/MS_R_50_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS BoxInst_MS_R_50_3x.pth
```

For training on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BoxInst/MS_R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/BoxInst_MS_R_50_1x
```

For evaluation on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BoxInst/MS_R_50_1x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/BoxInst_MS_R_50_1x \
    MODEL.WEIGHTS training_dir/BoxInst_MS_R_50_1x/model_final.pth
```

