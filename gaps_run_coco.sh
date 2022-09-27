DATA_ROOT=../data/COCO2017
DATASET=coco
TASK=15-1-split4
FOLDING=4
EPOCH=50 #50
BATCH=16 #32
LOSS=bce_loss
LR=0.01 # 0.01# try decreasing learning rate
THRESH=0.7
FEWSHOT=True
NUMSHOT=5
#NUMSHOT=1
MEMORY=500 # [0 (for SSUL), 100 (for SSUL-M)]

###### few shot step 1 - last
python gaps_main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
                    --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
                    --dataset ${DATASET} --task ${TASK} --folding ${FOLDING} --lr_policy poly --pseudo \
                    --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
                    --mem_size ${MEMORY} \
                    --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} \
                    --ckpt ./checkpoints/deeplabv3_resnet101_coco_15-1-split4_step_0_disjoint.pth \
