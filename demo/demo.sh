CUDA_VISIBLE_DEVICES=0 python demo.py \    
    --config-file ../configs/OREINST/oreinst.yaml \     
    --input /home/dlhuang/AdelaiDet-BoxInst/datasets/coco/val2017 \   
    --output /home/dlhuang/AdelaiDet-BoxInst/detection_results \  
    --opts MODEL.WEIGHTS \
    ../output/OREINST/model_0013999.pth