import warnings
import os
from global_config import Preprocess_config
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/heye0507/apex')
from apex import amp

import torch
from datetime import datetime
import time
from glob import glob

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



opt_level = Preprocess_config().opt_level # apex

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
                summary_loss.update(loss.detach().item()/self.iters_to_accumulate, batch_size)

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


# class Fitter:
    
#     def __init__(self, model, device, config):
#         self.config = config
#         self.epoch = 0

#         self.base_dir = f'/home/heye0507/wheat_detection/model/{config.folder}'
#         if not os.path.exists(self.base_dir):
#             os.makedirs(self.base_dir)
        
#         self.log_path = f'{self.base_dir}/log.txt'
#         self.best_summary_loss = 10**5

#         self.model = model
#         self.device = device

#         param_optimizer = list(self.model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ] 

#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        
#         self.model, self.optimizer = amp.initialize(self.model,self.optimizer,opt_level=opt_level) # apex
        
#         self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
#         self.log(f'Fitter prepared. Device is {self.device}')
        
        

#     def fit(self, train_loader, validation_loader):
#         for e in range(self.config.n_epochs):
#             if self.config.verbose:
#                 lr = self.optimizer.param_groups[0]['lr']
#                 timestamp = datetime.utcnow().isoformat()
#                 self.log(f'\n{timestamp}\nLR: {lr}')

#             t = time.time()
#             summary_loss = self.train_one_epoch(train_loader)

#             self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
#             self.save(f'{self.base_dir}/last-checkpoint.bin')

#             t = time.time()
#             summary_loss = self.validation(validation_loader)

#             self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
#             if summary_loss.avg < self.best_summary_loss:
#                 self.best_summary_loss = summary_loss.avg
#                 self.model.eval()
#                 self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
#                 for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
#                     os.remove(path)

#             if self.config.validation_scheduler:
#                 self.scheduler.step(metrics=summary_loss.avg)

#             self.epoch += 1

#     def validation(self, val_loader):
#         self.model.eval()
#         summary_loss = AverageMeter()
#         t = time.time()
#         for step, (images, targets, image_ids) in enumerate(val_loader):
#             if self.config.verbose:
#                 if step % self.config.verbose_step == 0:
#                     print(
#                         f'Val Step {step}/{len(val_loader)}, ' + \
#                         f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                         f'time: {(time.time() - t):.5f}', end='\r'
#                     )
#             with torch.no_grad():
#                 images = torch.stack(images)
#                 batch_size = images.shape[0]
#                 images = images.to(self.device).float()
#                 boxes = [target['bbox'].to(self.device).float() for target in targets]
#                 labels = [target['cls'].to(self.device).float() for target in targets]
                
#                 target_gt = {}
#                 target_gt['bbox'] = boxes
#                 target_gt['cls'] = labels

#                 output = self.model(images, target_gt)
#                 loss = output['loss']
#                 summary_loss.update(loss.detach().item(), batch_size)

#         return summary_loss

#     def train_one_epoch(self, train_loader):
#         self.model.train()
#         summary_loss = AverageMeter()
#         t = time.time()
#         for step, (images, targets, image_ids) in enumerate(train_loader):
#             if self.config.verbose:
#                 if step % self.config.verbose_step == 0:
#                     print(
#                         f'Train Step {step}/{len(train_loader)}, ' + \
#                         f'summary_loss: {summary_loss.avg:.5f}, ' + \
#                         f'time: {(time.time() - t):.5f}', end='\r'
#                     )
            
#             images = torch.stack(images)
#             images = images.to(self.device).float()
#             batch_size = images.shape[0]
#             boxes = [target['bbox'].to(self.device).float() for target in targets]
#             labels = [target['cls'].to(self.device).float() for target in targets]
            
#             target_gt = {}
#             target_gt['bbox'] = boxes
#             target_gt['cls'] = labels

#             self.optimizer.zero_grad()
            
#             output = self.model(images, target_gt)
#             loss = output['loss']
            
#             with amp.scale_loss(loss,self.optimizer) as scaled_loss: # apex
#                 scaled_loss.backward()
#             #loss.backward()

#             summary_loss.update(loss.detach().item(), batch_size)

#             self.optimizer.step()

#             if self.config.step_scheduler:
#                 self.scheduler.step()

#         return summary_loss
    
#     def save(self, path):
#         self.model.eval()
#         torch.save({
#             'model_state_dict': self.model.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'best_summary_loss': self.best_summary_loss,
#             'epoch': self.epoch,
#             'amp': amp.state_dict() # apex
#         }, path)

#     def load(self, path):
#         checkpoint = torch.load(path)
#         self.model.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         self.best_summary_loss = checkpoint['best_summary_loss']
#         self.epoch = checkpoint['epoch'] + 1
        
#     def log(self, message):
#         if self.config.verbose:
#             print(message)
#         with open(self.log_path, 'a+') as logger:
#             logger.write(f'{message}\n')
