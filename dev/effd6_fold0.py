import sys
sys.path.append('/home/heye0507/apex')
from apex import amp

import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob

from pathlib import Path

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


path = Path('/home/heye0507/wheat_detection/data')

df = pd.read_csv(path/'train.csv')

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    df[column] = bboxs[:,i]
df.drop(columns=['bbox'], inplace=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = df[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.Blur(blur_limit=3,p=0.2),
#             A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=1024, width=1024, p=1),
            #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
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

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1.0),
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
    
TRAIN_ROOT_PATH = path/'train'

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
        
        
fold_number = 0

train_dataset = Wheat_Dataset(
    df=df,
    path=str(path),
    image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    transform=get_train_transforms(),
)

validation_dataset = Wheat_Dataset(
    df=df,
    path=str(path),
    image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    transform=get_valid_transforms(),
    test=True,
)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
import warnings
warnings.filterwarnings("ignore")

opt_level ='O1' # apex

class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/home/heye0507/wheat_detection/model/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        
        self.model, self.optimizer = amp.initialize(self.model,self.optimizer,opt_level=opt_level) # apex
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        
        self.iters_to_accumulate = 4

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['bbox'].to(self.device).float() for target in targets]
                labels = [target['cls'].to(self.device).float() for target in targets]
                
                target_gt = {}
                target_gt['bbox'] = boxes
                target_gt['cls'] = labels

                output = self.model(images, target_gt)
                loss = output['loss']
                summary_loss.update(loss.detach().item()/4, batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['bbox'].to(self.device).float() for target in targets]
            labels = [target['cls'].to(self.device).float() for target in targets]
            
            target_gt = {}
            target_gt['bbox'] = boxes
            target_gt['cls'] = labels

            
            
            output = self.model(images, target_gt)
            loss = output['loss']
            
            loss = loss / self.iters_to_accumulate
            
            with amp.scale_loss(loss,self.optimizer) as scaled_loss: # apex
                scaled_loss.backward()
            #loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            
            if (step+1) % self.iters_to_accumulate == 0:

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.config.step_scheduler:
                    self.scheduler.step()

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            'amp': amp.state_dict() # apex
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
class TrainGlobalConfig:
    num_workers = 4
    batch_size = 2
    n_epochs = 80 # n_epochs = 40
    lr = 0.0002

    folder = 'effd6_fold0'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=1e-3,
        total_steps = len(train_dataset) // 4 * n_epochs,
        #epochs=n_epochs,
        #steps_per_epoch=int(len(train_dataset) / batch_size),
        pct_start=0.3,
        anneal_strategy='cos', 
        final_div_factor=10**5
    )
    
#     SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
#     scheduler_params = dict(
#         mode='min',
#         factor=0.5,
#         patience=1,
#         verbose=False, 
#         threshold=0.0001,
#         threshold_mode='abs',
#         cooldown=0, 
#         min_lr=1e-8,
#         eps=1e-08
#     )

def collate_fn(batch):
    return tuple(zip(*batch))

def run_training():
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)
    

import torch
import torch.nn as nn
from timm.utils import ModelEma
from effdet.anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from effdet.loss import DetectionLoss

class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(self.anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, target):
        class_out, box_out = self.model(x)
        cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
            x.shape[0], target['bbox'], target['cls'])
        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        output = dict(loss=loss, class_loss=class_loss, box_loss=box_loss)
#         if not self.training:
#             # if eval mode, output detections for evaluation
#             class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
#             output['detections'] = _batch_detection(
#                 x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes,
#                 target['img_scale'], target['img_size'])
        return output
        
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet

def get_net():
    config = get_efficientdet_config('tf_efficientdet_d6')
    net = EfficientDet(config, pretrained_backbone=True)
    config.num_classes = 1
    config.image_size = 1024
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

net = get_net()

run_training()
