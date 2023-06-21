#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import pandas as pd
import numpy as np
import experiment_variables as ev
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import accuracy
import cv2
import math
### Dataset parameters
EPSILON = ev.EPSILON
annotation = ev.annotation
dimension = ev.dimension
WIN_SIZE = ev.WIN_SIZE
STEP = ev.STEP
F_SKIP = ev.F_SKIP
RESCALE_FACTOR = ev.RESCALE_FACTOR
arousal_thres = ev.arousal_thres
valence_thres=ev.valence_thres

### Model parameters
input_channels = ev.INPUT_CHANNELS
batch_sz = ev.BATCH_SZ
def classification_data(filename,minmax=True):
    data = pd.read_csv(filename)
    data[dimension+annotation+" sample"] -= arousal_thres if "arousal" in dimension else valence_thres # to make the classification problem more balanced
    data_classification = data[np.abs(data[dimension+annotation+" sample"]) > EPSILON]
    if minmax:
        for col in data_classification.columns:
            if len(data_classification[col].unique()) == 1:
                data_classification.drop(col, inplace=True, axis=1)
        #modify indices accordingly
        features = copy.deepcopy(data_classification.iloc[:, 0:289])
        temp =  MinMaxScaler((-1,1)).fit_transform(features)
        
        data_classification.iloc[:, 0:289] = temp
        
    return data_classification, np.asarray(data_classification['participant id'])


class ClassificationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # raw images
        num_of_frames = int(WIN_SIZE / F_SKIP)  
        concatenated_imgs = [] 
        for i in range(num_of_frames):
            img_name = self.data[str(i)].iloc[idx][:-4]
            # img = io.imread(img_name)
            # gray_img = rgb2gray(img)
            # gray_img = rescale(gray_img, RESCALE_FACTOR, anti_aliasing=True)
            gray_img = cv2.imread(img_name+'.jpg')
            gray_img= cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
            # gray_img = np.random.uniform(size=(180,320))
            concatenated_imgs.append(gray_img)
        concatenated_imgs = np.asarray(concatenated_imgs)/255
        
        # audio features
        audio_columns = [col for col in self.data.columns if 'ComParE13_LLD' in col or 'audio_speech_probability' in col]
        audio_features = np.asarray(self.data[audio_columns].iloc[idx])
        
        # video features
        video_columns = [col for col in self.data.columns if 'VIDEO_40_LLD' in col or 'Face_detection_probability' in col]
        video_features = np.asarray(self.data[video_columns].iloc[idx])
        
        # ECG features
        ecg_columns = [col for col in self.data.columns if 'ECG_54_LLD' in col]
        ecg_features = np.asarray(self.data[ecg_columns].iloc[idx])
        
        # EDA features
        eda_columns = [col for col in self.data.columns if 'EDA_62_LLD' in col]
        eda_features = np.asarray(self.data[eda_columns].iloc[idx])
        
        # binary labels
        label_continuous = self.data[dimension+annotation+' sample'].iloc[idx]
        
        time = self.data['time'].iloc[idx]
        
        if label_continuous >= 0:
            label = 1
        else:
            label = 0
                 
        sample = {'images': concatenated_imgs,
                  'audio': audio_features,
                  'video': video_features,
                  'ECG': ecg_features,
                  'EDA': eda_features,
                  'label': np.asarray(label, dtype=np.int64)}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ToTensor(object):
    def  __call__(self, sample):
        images = sample['images'].astype('float32')
        audio = sample['audio'].astype('float32')
        video = sample['video'].astype('float32')
        ecg = sample['ECG'].astype('float32')
        eda = sample['EDA'].astype('float32')
        label = sample['label']
        
        return {'images': torch.from_numpy(images),
                'audio': torch.from_numpy(audio),
                'video': torch.from_numpy(video),
                'ECG': torch.from_numpy(ecg),
                'EDA': torch.from_numpy(eda),
                'label': torch.from_numpy(label.reshape(-1,))}
        
        
############################### Models ########################################
class CnnImage(LightningModule):
    def __init__(self, train_data, val_data, test_data):
        super().__init__()
        self.emb=nn.Linear(512, 96)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(288, 96)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(96,2)
        
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)

    def training_step(self, batch, batch_idx):
        x, y = batch['images'], batch['label'].reshape((-1))
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'loss':loss, 'train_acc':acc}
    
    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        print('Training Accuracy: {:.4f}'.format(avg_acc))
        self.train_acc.append(avg_acc.item())
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['images'], val_batch['label'].reshape((-1))
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {'val_acc':acc, 'val_loss':loss}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation Accuracy: {:.4f}'.format(avg_acc))
        self.val_acc.append(avg_acc.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['images'], test_batch['label'].reshape((-1))
        logits = self(x)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'test_acc':acc}
    
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Testing Accuracy: {:.4f}'.format(avg_acc))
        self.test_acc.append(avg_acc.item())
        #return {'avg_test_acc':avg_acc}
        
        
class FusionModel(LightningModule):
    def __init__(self, train_data, val_data, test_data):
        super().__init__()
        self.emb = nn.Linear(512,2*96)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc1_img = nn.Linear(288, 96)
        self.fc1_priv = nn.Linear(289, 96)
        self.dropout = nn.Dropout(0.1)
        self.fc_reduce = nn.Linear(2*96,96)
        self.fc2 = nn.Linear(96,2)
        
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        
        
    def forward(self, x_img, x_priv):
        x_img = F.relu(self.conv1(x_img))
        x_img = self.pool1(x_img)
        x_img = F.relu(self.conv2(x_img))
        x_img = self.pool2(x_img)
        x_img = F.relu(self.conv3(x_img))
        x_img = self.pool3(x_img)
        x_img = F.relu(self.conv4(x_img))
        x_img = self.pool4(x_img)
        x_img = x_img.view(-1, 288)
        x_img = F.relu(self.fc1_img(x_img))
        x_priv = F.relu(self.fc1_priv(x_priv))
        x = torch.cat((x_img, x_priv), axis = 1)  
        x = F.relu(self.fc_reduce(x)) 
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        x_img, y = batch['images'], batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        logits = self(x_img, x_priv)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'loss':loss, 'train_acc':acc}
    
    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        print('Training Accuracy: {:.4f}'.format(avg_acc))
        self.train_acc.append(avg_acc.item())
    
    def validation_step(self, val_batch, batch_idx):
        x_img, y = val_batch['images'], val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)  
        logits = self(x_img, x_priv)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {'val_acc':acc, 'val_loss':loss}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation Accuracy: {:.4f}'.format(avg_acc))
        self.val_acc.append(avg_acc.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x_img, y = test_batch['images'], test_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = test_batch['audio'], test_batch['EDA'], test_batch['ECG'], test_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        logits = self(x_img, x_priv)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'test_acc':acc}
    
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Testing Accuracy: {:.4f}'.format(avg_acc))
        self.test_acc.append(avg_acc.item())
    

           
class TeacherModel(LightningModule):
    def __init__(self, train_data, val_data, test_data):
        super().__init__()
    
        self.fc1_priv = nn.Linear(289, 96)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(96,2)
        
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        
        
    def forward(self, x_img, x_priv):        
        x_priv = F.relu(self.fc1_priv(x_priv))
        x_priv = self.dropout(x_priv)
        x_priv = self.fc2(x_priv)
        
        return x_priv
    
    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        y = batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1) 
        x_img=None
        logits = self(x_img,x_priv)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'loss':loss, 'train_acc':acc}
    
    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        print('Training Accuracy: {:.4f}'.format(avg_acc))
        self.train_acc.append(avg_acc.item())
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        logits = self.forward(x_img,x_priv)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {'val_acc':acc, 'val_loss':loss}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation Accuracy: {:.4f}'.format(avg_acc))
        self.val_acc.append(avg_acc.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        y = test_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = test_batch['audio'], test_batch['EDA'], test_batch['ECG'], test_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        logits = self.forward(x_img,x_priv)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'test_acc':acc}
    
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Testing Accuracy: {:.4f}'.format(avg_acc))
        self.test_acc.append(avg_acc.item())
        

class StudentModel(LightningModule):
    def __init__(self, train_data, val_data, test_data, teacher, alpha):
        super().__init__()
        
    
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.fc1_img = nn.Linear(288, 96)
        self.dropout = nn.Dropout(0.1)
        self.fc2_img = nn.Linear(96, 2)
        
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.alpha = alpha
        
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        

        
    def forward(self, x_img):
        x_img = F.relu(self.conv1(x_img))
        x_img = self.pool1(x_img)
        x_img = F.relu(self.conv2(x_img))
        x_img = self.pool2(x_img)
        x_img = F.relu(self.conv3(x_img))
        x_img = self.pool3(x_img)
        x_img = F.relu(self.conv4(x_img))
        x_img = self.pool4(x_img)
        x_img = x_img.view(-1, 288)
        x_img = F.relu(self.fc1_img(x_img))
        x_img = self.dropout(x_img)
        x_img = self.fc2_img(x_img)
        
        return x_img
    
    def cross_entropy_loss(self, logits, teacher_logits, labels):
        T = 0.5
        
        classification_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        class_loss = (1.0 - self.alpha) * classification_criterion(logits, labels)
        
        soft_logits = F.log_softmax(logits/T, dim=1)
        soft_teacher_logits = F.softmax(teacher_logits/T, dim=1)
        
        privileged_criterion = nn.KLDivLoss(reduction= 'batchmean')
        priv_loss = self.alpha * privileged_criterion(soft_logits, soft_teacher_logits)
        
        return priv_loss + class_loss


    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        x_img, y = batch['images'], batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv_teacher = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        logits=self(x_img)
        teacher_logits = self.teacher(x_img, x_priv_teacher) # teacher's predictions         
        loss = self.cross_entropy_loss(logits, teacher_logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'loss':loss, 'train_acc':acc}
    
    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        print('Training Accuracy: {:.4f}'.format(avg_acc))
        self.train_acc.append(avg_acc.item())
    
    def validation_step(self, val_batch, batch_idx):
        x_img, y = val_batch['images'], val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv_teacher = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        logits=self(x_img)
        teacher_logits = self.teacher(x_img, x_priv_teacher) # teacher's predictions
        
        loss = self.cross_entropy_loss(logits, teacher_logits, y)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {'val_acc':acc, 'val_loss':loss}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Validation Accuracy: {:.4f}'.format(avg_acc))
        self.val_acc.append(avg_acc.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x_img, y = test_batch['images'], test_batch['label'].reshape((-1))
        logits = self.forward(x_img)
        _, preds = torch.max(logits, dim=1)
        acc = accuracy(preds, y, num_classes=2)
        return {'test_acc':acc}
    
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Testing Accuracy: {:.4f}'.format(avg_acc))
        self.test_acc.append(avg_acc.item())
    
    
