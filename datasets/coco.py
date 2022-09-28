import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .utils import Subset #, filter_images, group_images
import torch
import json

from PIL import Image
import torchvision.transforms as T

from .coco_base import COCOSeg
from .coco_20i import COCO20iReader
from .baseset import base_set
from utils import tasks
import torchvision.transforms.functional as F

class COCOSegmentation(data.Dataset):
    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None,
                 cil_step=0,
                 mem_size=0,
                 seed=2022):

        self.opts = opts
        self.root = opts.data_root
        self.task = opts.task
        self.overlap = opts.overlap
        self.unknown = opts.unknown

        self.transform = transform
        self.image_set = image_set

        self.folding = opts.folding
        self.few_shot = opts.few_shot
        self.num_shot = opts.num_shot

        # COCO_PATH = os.path.join(self.root, "COCO2017")
        COCO_PATH = self.root
        self.target_cls = tasks.get_tasks('coco', self.task, cil_step)

        self.target_cls += [255]  # including ignore index (255)

        if cil_step == 0:
            if image_set == 'train':
                ds = COCO20iReader(COCO_PATH, self.folding, True, exclude_novel=True)
                self.dataset = base_set(ds, "train")
            else:
                ds = COCO20iReader(COCO_PATH, self.folding, False, exclude_novel=False)
                self.dataset = base_set(ds, "test")
        else:
            if image_set == 'train':
                ds = COCOSeg(COCO_PATH, True)
                dataset = base_set(ds, "test")  # Use test config to keep original scale of the image.

                idxs = list(range(len(ds)))
                final_index_list = []
                self.target_cls.remove(255)
                if self.few_shot:
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    for k in self.target_cls:
                        label_idxs = ds.get_class_map(k)
                        for _ in range(self.num_shot):
                            idx = random.choice(label_idxs)
                            while True:
                                novel_img_chw, mask_hw, _ = dataset[idx]
                                pixel_sum = torch.sum(mask_hw == k)
                                # If the selected sample is bad (more than 1px) and has not been selected,
                                # we choose the example.
                                if pixel_sum > 1 and idx not in final_index_list:
                                    final_index_list.append(idx)
                                    break
                                else:
                                    idx = random.choice(label_idxs)
                else:
                    final_index_list = idxs

                self.target_cls += [255]

                idxs = final_index_list

                ###################################################################
                if self.num_shot == 5:
                    idxs = idxs * 20
                elif self.num_shot == 1:
                    idxs = idxs * 100
                ###################################################################

                self.dataset = Subset(dataset, idxs)
            elif image_set == 'memory':
                for s in range(cil_step):
                    self.target_cls += tasks.get_tasks('coco', self.task, s)

                coco_root = './datasets/data/coco'
                memory_json = os.path.join(coco_root, 'memory.json')

                with open(memory_json, "r") as json_file:
                    memory_list = json.load(json_file)

                file_idxs = memory_list[f"step_{cil_step}"]["memory_list"]
                print("... memory list : ", len(file_idxs), self.target_cls)

                while len(file_idxs) < opts.batch_size:
                    file_idxs = file_idxs * 2

                ds = COCOSeg(COCO_PATH, True)
                dataset = base_set(ds, "test")
                self.dataset = Subset(dataset, file_idxs)
            else:
                ds = COCOSeg(COCO_PATH, False)
                self.dataset = base_set(ds, "test")

        # class re-ordering
        all_steps = tasks.get_tasks('coco', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]

        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in
                                                range(len(all_classes))]

        ###########################################################
        # if self.image_set == 'memory':
        #     memory_class_occur = []
        #     for i in range(len(self.dataset)):
        #         img_chw, mask_hw, _ = self.dataset[i]
        #         target = mask_hw.type(torch.float)
        #         target = self.gt_label_mapping(target)
        #
        #         target = torch.from_numpy(np.array(target, dtype='uint8'))
        #
        #         target = target.long()
        #         cur_set = set([i.item() for i in torch.unique(target)])
        #         memory_class_occur.extend(cur_set)
        #     memory_class_occur = set(memory_class_occur)
        #     print('print memory classes occured in coco.py: ')
        #     print(memory_class_occur)
    ###########################################################


    def __getitem__(self, index):
        img, target, file_id = self.dataset[index]

        target = target.type(torch.float)

        sal_map = Image.fromarray(np.ones(target.size()[::-1], dtype=np.uint8))

        # re-define target label according to the CIL case
        target = self.gt_label_mapping(target)

        if self.transform is not None:
            img, target, sal_map = self.transform(img, target, sal_map)


        # add unknown label, background index: 0 -> 1, unknown index: 0
        # if self.image_set == 'train' and self.unknown:
        if (self.image_set == 'train' or self.image_set == 'memory') and self.unknown:
            target = torch.where(target == 255,
                                 torch.zeros_like(target) + 255,  # keep 255 (uint8)
                                 target + 1)  # unknown label

            unknown_area = (target == 1)
            target = torch.where(unknown_area, torch.zeros_like(target), target)

        return img, target.long(), sal_map, file_id


    def __len__(self):
        return len(self.dataset)

    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)


        if self.image_set != 'test':
            # gt = np.where(True, gt, 0)
            gt = np.where(np.isin(gt, self.target_cls), gt, 0)

        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)

        return gt

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
