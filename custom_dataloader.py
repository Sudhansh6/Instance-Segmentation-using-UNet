import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms as T

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, filter_classes, size=None, max_num=None):
        self.image_dir = image_dir
        self.filter_classes = filter_classes
        self.coco = COCO(annotation_file)
        self.coco_mask = mask 

        # Fetch class IDs only corresponding to the filterClasses
        self.catIds = self.coco.getCatIds(catNms=filter_classes)

        # Get image ID that satisfies the given filter conditions
        self.ids = self.coco.getImgIds()
        print("Loaded", len(self.ids), "images")
        
        if max_num != None:
            self.ids = self.ids[:max_num]

        self.size = size
        if size != None:
            self.transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        mask = np.zeros(image.size).T
       
        # 0 is background
        for ann in anns:
            print(ann.keys())
            idx = self.catIds.index(ann['category_id']) + 1
            mask = np.maximum(mask, \
                                self.coco.annToMask(ann)*idx)

        mask = Image.fromarray(mask)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

if __name__ == "__main__":
    dataDir='./coco-2017/'
    dataType='train2017'
    annFile='{}raw/instances_{}.json'.format(dataDir, dataType)
    imageDir = '{}train/data/'.format(dataDir)
    filter_classes = ['person', 'cat', 'car']


    # Define transformations
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    class pairTransform(torch.nn.Module):
        def __init__(self, transforms):
            super().__init__()
            self.transforms = transforms

        def __call__(self, imgs):
            return [self.transforms(img) for img in imgs]

    pairTransformo = pairTransform(transform)

    size = 128
    # Initialize the COCO api for instance annotations
    coco_dataset = COCODataset(imageDir, annFile, filter_classes, size)

    for i, m in coco_dataset:
        break