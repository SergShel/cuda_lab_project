import os
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

# code snippets from https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5

class Coco_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, sizes=(256, 256), mode="segmentation"):
        self.root = root
        # Transforms are hard coded so far
        # self.transforms = transforms
        # self.target_transforms = target_transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        if sizes != None:
            resize = T.Compose([
                T.Resize(sizes, interpolation= T.InterpolationMode.BILINEAR),
                T.ToTensor()
            ])

            target_resize = T.Compose([
                T.Resize(sizes, interpolation= T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])

            self.transforms = resize
            self.target_transforms = target_resize

            # to get segmentation
            self.mode = mode

        

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        if self.transforms != None:
            img = self.transforms(img)

        # print("img.shape in getitem ", img.shape)

        if img.shape[0] == 1:
            # Convert grayscale image to RGB
            img = torch.stack([img[0]] * 3, dim=0)



        # create segmentation mask
        seg_mask = torch.zeros((1, img.shape[1], img.shape[2]))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # get masks and according labels 
        masks, labels = [], []
        for i in range(num_objs):
            obj_mask = coco.annToMask(ann=coco_annotation[i])
            # transform mask:
            if self.target_transforms != None:
                # transform to PIL type for a resize 
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = self.target_transforms(obj_mask) 
            
            label = coco_annotation[i]['category_id']
            # Because of to Tensor the obj_mask is not of values 0 and 1 anymore
            # -> get new mask and replace pixel values with labels
            mask = (obj_mask > 0)
            seg_mask[mask] = label
            # Conflict: Adding labels together results to mixup of labels
            # seg_mask = seg_mask + (mask * 255) * label

            masks.append(mask)
            labels.append(label)
            
        labels = torch.tensor(labels)
                
        # Annotation is in dictionary format
        label_dict = {}
        label_dict["segmentation_mask"] = seg_mask
        label_dict["masks"] = masks
        label_dict["labels"] = labels


        return (img, seg_mask) if self.mode=="segmentation" else (img, label_dict)

    def __len__(self):
        return len(self.ids)