import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch


class Datapipline():
    def __init__(self,path,imsize=1024):
        '''
            Args:
                path is the root path
                path/'train.csv' will be read
                path/'data' will be load as image file path
        '''
        self.path = path
        self.imsize = imsize
        self._prepare_fold()
        
    def _prepare_fold(self):
        self.df = pd.read_csv(self.path+'/train.csv')
        bboxs = np.stack(self.df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            self.df[column] = bboxs[:,i]
        self.df.drop(columns=['bbox'], inplace=True)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.df_folds = self.df[['image_id']].copy()
        self.df_folds.loc[:, 'bbox_count'] = 1
        self.df_folds = self.df_folds.groupby('image_id').count()
        self.df_folds.loc[:, 'source'] = self.df[['image_id', 'source']].groupby('image_id').min()['source']
        self.df_folds.loc[:, 'stratify_group'] = np.char.add(
            self.df_folds['source'].values.astype(str),
            self.df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
        )
        self.df_folds.loc[:, 'fold'] = 0

        for fold_number, (train_index, val_index) in enumerate(skf.split(X=self.df_folds.index, y=self.df_folds['stratify_group'])):
            self.df_folds.loc[self.df_folds.iloc[val_index].index, 'fold'] = fold_number
            
    def get_train_transforms(self):
        return A.Compose(
            [
                A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                         val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2, 
                                               contrast_limit=0.2, p=0.9),
                ],p=0.9),
                #A.Blur(blur_limit=3,p=0.2),
    #             A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=self.imsize, width=self.imsize, p=1),
                A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                A.Normalize(p=1.0,max_pixel_value=1.0),
                ToTensorV2(p=1.0),
            ], 
            p=1.0, 
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0, 
                min_visibility=0,
                label_fields=['labels']
            )
        )

    def get_valid_transforms(self):
        return A.Compose(
            [
                A.Resize(height=self.imsize, width=self.imsize, p=1.0),
                A.Normalize(p=1.0,max_pixel_value=1.0),
                ToTensorV2(p=1.0),
            ], 
            p=1.0, 
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0, 
                min_visibility=0,
                label_fields=['labels']
            )
        )

class Wheat_Dataset(Dataset):
    def __init__(self,df,path,image_ids,transform=None,test=False,yxyx=True):
        super().__init__()
        
        self.df = df
        self.base_path = path if path[-1] != '/' else path[:-1]
        self.image_ids = image_ids
        self.transform =transform
        self.test = test
        self.yxyx = yxyx # to algin with google effdet
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,idx):
        image_id = self.image_ids[idx]
        
        if not self.test and random.randint(0,1) < 0.5:
            image,boxes = self.load_cutmix_all(idx)
        else:
            image,boxes = self.load_image_and_boxes(idx)
        
        labels = torch.ones((boxes.shape[0]))
        
        if self.transform:
            while True: # avoid zoom get empty bbox
                aug = self.transform(**{
                    'image':image,
                    'bboxes':boxes,
                    'labels':labels
                })
                if len(aug['bboxes']) > 0:
                    break
                    
        
        target = {}
        target['bbox'] = torch.tensor(aug['bboxes'])
        if self.yxyx:
            target['bbox'][:,[0,1,2,3]] = target['bbox'][:,[1,0,3,2]]
        target['cls'] = torch.tensor(aug['labels'])
        
        image = aug['image']
        
        return image,target,image_id
        
        
    
    def load_image_and_boxes(self,idx):
        image_id = self.image_ids[idx]
        file_path = self.base_path + '/train/' + image_id + '.jpg'
        image = cv2.imread(file_path,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.
        
        # get box
        boxes = self.df[self.df['image_id']==image_id][['x','y','w','h']].values
        boxes[:,2] = boxes[:,2] + boxes[:,0]
        boxes[:,3] = boxes[:,3] + boxes[:,1]
        
        return image, boxes
    
    def load_cutmix_image_and_boxes(self,idx,imsize=1024):
        img_1,boxes_1 = self.load_image_and_boxes(idx)
        while True:
            idx_2 = random.randint(0,len(self.image_ids)-1)
            if idx_2 != idx:
                break
        img_2,boxes_2 = self.load_image_and_boxes(idx_2)
        
        # setup position tp cut for img_1 and img_2
        xc,yc = [int(random.uniform(imsize*0.4,imsize*0.6)) for _ in range(2)]
        w,h = imsize,imsize
        pos = random.randint(0,1)
        if pos == 0: #top left
            x1a,y1a,x2a,y2a = 0,0,xc,yc # img_1
            x1b,y1b,x2b,y2b = w-xc,h-yc,w,h # img_2
        elif pos == 1: # top right
            x1a,y1a,x2a,y2a = w-xc,0,w,yc
            x1b,y1b,x2b,y2b = 0,h-yc,xc,h
        elif pos == 2: # bottom left
            x1a,y1a,x2a,y2a = 0,h-yc,xc,h
            x1b,y1b,x2b,y2b = w-xc,0,w,yc
        elif pos == 3: # bottom right
            x1a,y1a,x2a,y2a = w-xc,h-yc,w,h
            x1b,y1b,x2b,y2b = 0,0,xc,yc
        
        # create result img
        mixup_image = img_1.copy() 
        mixup_image[y1a:y2a,x1a:x2a] = (mixup_image[y1a:y2a,x1a:x2a] + img_2[y1b:y2b,x1b:x2b])/2
        
        result_boxes = []
        result_boxes.append(boxes_2)
        result_boxes = np.concatenate(result_boxes, 0)

        padw = x1a-x1b
        pady = y1a-y1b

        result_boxes[:,0] += padw
        result_boxes[:,1] += pady
        result_boxes[:,2] += padw
        result_boxes[:,3] += pady

        np.clip(result_boxes,0,imsize,out=result_boxes)
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        mixup_image_boxes = np.concatenate((boxes_1,result_boxes),axis=0)
        
        return mixup_image, mixup_image_boxes
    
    def load_mixup_image_and_boxes(self,idx):
        img_1,boxes_1 = self.load_image_and_boxes(idx)
        while True:
            idx_2 = random.randint(0,len(self.image_ids)-1)
            if idx_2 != idx:
                break
        img_2,boxes_2 = self.load_image_and_boxes(idx_2)
        mixup_img = (img_1+img_2)/2
        
        mixup_img_boxes = np.concatenate((boxes_1,boxes_2),axis=0)
        
        return mixup_img,mixup_img_boxes
    
    def load_cutmix_all(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes
