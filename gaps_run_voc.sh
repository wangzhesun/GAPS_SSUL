DATA_ROOT=../data/VOCdevkit/VOC2012
DATASET=voc
TASK=15-1-split3
EPOCH=50
BATCH=16
LOSS=bce_loss
LR=0.01
THRESH=0.7
FEWSHOT=True
#NUMSHOT=5
NUMSHOT=1
MEMORY=500 # 300 # [0 (for SSUL), 100 (for SSUL-M)]

python3 gaps_main.py --crop_val --data_root ${DATA_ROOT} --model deeplabv3_resnet101 \
        --gpu_id 0 --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
        --dataset ${DATASET} --task ${TASK} --lr_policy poly --pseudo --pseudo_thresh ${THRESH} \
        --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY} \
        --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} \
        --ckpt ./checkpoints/deeplabv3_resnet101_voc_15-1-split3_step_0_disjoint.pth \

############  DO NOT forget to change the class old in tasks.py
