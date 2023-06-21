#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pandas as pd
import numpy as np
import cv2
import experiment_variables as ev
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import r2_score, pearson_corrcoef
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

def regression_data(filename,minmax=True):
    data_regression = pd.read_csv(filename)
   
    if minmax:
        for col in data_regression.columns:
            if len(data_regression[col].unique()) == 1:
                data_regression.drop(col, inplace=True, axis=1)
        #modify indices accordingly            
        features = copy.deepcopy(data_regression.iloc[:, 0:289])
        temp =  MinMaxScaler((-1,1)).fit_transform(features)
        
        data_regression.iloc[:, 0:289] = temp
    
    return data_regression, np.asarray(data_regression['participant id'])


class RegressionDataset(Dataset):
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
            
            gray_img = cv2.imread(img_name+'.jpg')
            gray_img= cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
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
        
        # continuous labels
        label = self.data[dimension+annotation+' sample'].iloc[idx]

        
        # participant ids
        p_ids = self.data['participant id'].iloc[idx]
                 
        sample = {'images': concatenated_imgs,
                  'audio': audio_features,
                  'video': video_features,
                  'ECG': ecg_features,
                  'EDA': eda_features,
                  'label': np.asarray(label, dtype=np.float32),
                  'p_ids': np.asarray(p_ids,  dtype=np.int64)}
        
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
        p_ids = sample['p_ids']
        
        return {'images': torch.from_numpy(images),
                'audio': torch.from_numpy(audio),
                'video': torch.from_numpy(video),
                'ECG': torch.from_numpy(ecg),
                'EDA': torch.from_numpy(eda),
                'label': torch.from_numpy(label.reshape(-1,)),
                'p_ids': torch.from_numpy(p_ids.reshape(-1,))}
        
        
############################### Models ########################################
class CnnImage(LightningModule):
    def __init__(self, train_data, val_data, test_data):
        super().__init__()
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
        self.fc2 = nn.Linear(96,1)
        self.train_r2 = []
        self.val_r2 = []
        self.test_r2 = []
        
        self.train_mse = []
        self.val_mse = []
        self.test_mse = []
        
        self.train_rho = []
        self.val_rho = []
        self.test_rho = []
        
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
    
    def rmsi_loss(self, n_outputs, labels):
        criterion = nn.MSELoss()
        return criterion(n_outputs, labels)
    
    def train_dataloader(self, s=True):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=s)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['images'], batch['label'].reshape((-1))
        n_outputs = self(x).squeeze()
        return n_outputs

    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        x, y = batch['images'], batch['label'].reshape((-1))
        n_outputs = self(x).squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        return {'loss':loss, 'train_r2':r2, 'train_rho':rho}
    
    def training_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
        print('Training r2: {:.4f}'.format(avg_r2))
        self.train_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['train_rho'] for x in outputs]).mean()
        print('Training rho: {:.4f}'.format(avg_rho))
        self.train_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['loss'] for x in outputs]).mean()
        print('Training mse: {:.4f}'.format(avg_mse))
        self.train_mse.append(avg_mse.item())
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['images'], val_batch['label'].reshape((-1))
        n_outputs = self(x).squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        self.log("val_loss", loss)
        # self.log("val_r2", r2)
        return {'val_r2':r2, 'val_loss':loss, 'val_rho':rho}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
        print('Validation r2: {:.4f}'.format(avg_r2))
        self.val_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['val_rho'] for x in outputs]).mean()
        print('Validation rho: {:.4f}'.format(avg_rho))
        self.val_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation mse: {:.4f}'.format(avg_mse))
        self.val_mse.append(avg_mse.item())
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['images'], test_batch['label'].reshape((-1))
        n_outputs = self(x).squeeze()
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        mse = self.rmsi_loss(n_outputs, y)
        return {'test_r2':r2, 'test_rho':rho, 'test_mse':mse}
    
    def test_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['test_r2'] for x in outputs]).mean()
        print('Testing r2: {:.4f}'.format(avg_r2))
        self.test_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['test_rho'] for x in outputs]).mean()
        print('Testing rho: {:.4f}'.format(avg_rho))
        self.test_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        print('Testing mse: {:.4f}'.format(avg_mse))
        self.test_mse.append(avg_mse.item())
        
        
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
        self.fc2 = nn.Linear(96,1)
        
        self.train_r2 = []
        self.val_r2 = []
        self.test_r2 = []
        
        self.train_mse = []
        self.val_mse = []
        self.test_mse = []
        
        self.train_rho = []
        self.val_rho = []
        self.test_rho = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        
        
    def forward(self, x_img, x_priv,t):
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
        x1 = torch.cat((x_img, x_priv), axis = 1)       
        x1 = F.relu(self.fc_reduce(x1))
        x = self.dropout(x1)
        x = self.fc2(x)
        
        return x, x1
    
    def rmsi_loss(self, n_outputs, labels):
        criterion = nn.MSELoss()
        return criterion(n_outputs, labels)
    
    def train_dataloader(self, s=True):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=s)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['images'], batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, _ = self(x, x_priv)
        n_outputs = n_outputs.squeeze()
        return n_outputs
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        x_img, y = batch['images'], batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, _ = self(x_img, x_priv)
        n_outputs = n_outputs.squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        return {'loss':loss, 'train_r2':r2, 'train_rho':rho}
    
    def training_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
        print('Training r2: {:.4f}'.format(avg_r2))
        self.train_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['train_rho'] for x in outputs]).mean()
        print('Training rho: {:.4f}'.format(avg_rho))
        self.train_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['loss'] for x in outputs]).mean()
        print('Training mse: {:.4f}'.format(avg_mse))
        self.train_mse.append(avg_mse.item())
    
    def validation_step(self, val_batch, batch_idx):
        x_img, y = val_batch['images'], val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, _ = self(x_img, x_priv)
        n_outputs = n_outputs.squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        self.log("val_loss", loss)
        self.log("val_r2", r2)
        return {'val_r2':r2, 'val_loss':loss, 'val_rho':rho}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
        print('Validation r2: {:.4f}'.format(avg_r2))
        self.val_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['val_rho'] for x in outputs]).mean()
        print('Validation rho: {:.4f}'.format(avg_rho))
        self.val_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation mse: {:.4f}'.format(avg_mse))
        self.val_mse.append(avg_mse.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x_img, y = test_batch['images'], test_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = test_batch['audio'], test_batch['EDA'], test_batch['ECG'], test_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, _ = self(x_img, x_priv)
        n_outputs = n_outputs.squeeze()
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        mse = self.rmsi_loss(n_outputs, y)
        return {'test_r2':r2, 'test_rho':rho, 'test_mse':mse}
    
    def test_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['test_r2'] for x in outputs]).mean()
        print('Testing r2: {:.4f}'.format(avg_r2))
        self.test_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['test_rho'] for x in outputs]).mean()
        print('Testing rho: {:.4f}'.format(avg_rho))
        self.test_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        print('Testing mse: {:.4f}'.format(avg_mse))
        self.test_mse.append(avg_mse.item())
        
  
class TeacherModel(LightningModule):
    def __init__(self, train_data, val_data, test_data):
        super().__init__()
    
        self.fc1_priv = nn.Linear(289, 96)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(96,1)
        
        self.train_r2 = []
        self.val_r2 = []
        self.test_r2 = []
        
        self.train_mse = []
        self.val_mse = []
        self.test_mse = []
        
        self.train_rho = []
        self.val_rho = []
        self.test_rho = []
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
                        
        
    def forward(self,x_img, x_priv):        
        x1 = F.relu(self.fc1_priv(x_priv))
        x_priv = self.dropout(x1)
        x_priv = self.fc2(x_priv)
        
        return x_priv, x1
    
    def rmsi_loss(self, n_outputs, labels):
        criterion = nn.MSELoss()
        return criterion(n_outputs, labels)
    
    def train_dataloader(self, s=True):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=s)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        n_outputs, _ = self(x_img, x_priv)
        n_outputs = n_outputs.squeeze()
        return n_outputs
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        y = batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        n_outputs, _ = self(x_img,x_priv)
        n_outputs = n_outputs.squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        return {'loss':loss, 'train_r2':r2, 'train_rho':rho}
        
    
    def training_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
        print('Training r2: {:.4f}'.format(avg_r2))
        self.train_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['train_rho'] for x in outputs]).mean()
        print('Training rho: {:.4f}'.format(avg_rho))
        self.train_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['loss'] for x in outputs]).mean()
        print('Training mse: {:.4f}'.format(avg_mse))
        self.train_mse.append(avg_mse.item())
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        n_outputs, _ = self.forward(x_img,x_priv)
        n_outputs = n_outputs.squeeze()
        loss = self.rmsi_loss(n_outputs, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        self.log("val_loss", loss)
        self.log("val_r2", r2)
        return {'val_r2':r2, 'val_loss':loss, 'val_rho':rho}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
        print('Validation r2: {:.4f}'.format(avg_r2))
        self.val_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['val_rho'] for x in outputs]).mean()
        print('Validation rho: {:.4f}'.format(avg_rho))
        self.val_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation mse: {:.4f}'.format(avg_mse))
        self.val_mse.append(avg_mse.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        y = test_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = test_batch['audio'], test_batch['EDA'], test_batch['ECG'], test_batch['video']
        x_priv = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        x_img=None
        n_outputs, _ = self.forward(x_img,x_priv)
        n_outputs = n_outputs.squeeze()
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        mse = self.rmsi_loss(n_outputs, y)
        return {'test_r2':r2, 'test_rho':rho, 'test_mse':mse}
    
    def test_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['test_r2'] for x in outputs]).mean()
        print('Testing r2: {:.4f}'.format(avg_r2))
        self.test_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['test_rho'] for x in outputs]).mean()
        print('Testing rho: {:.4f}'.format(avg_rho))
        self.test_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        print('Testing mse: {:.4f}'.format(avg_mse))
        self.test_mse.append(avg_mse.item())

        

        
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
        self.fc2_img = nn.Linear(96, 1)
        
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.alpha = alpha
        
        self.train_r2 = []
        self.val_r2 = []
        self.test_r2 = []
        
        self.train_mse = []
        self.val_mse = []
        self.test_mse = []
        
        self.train_rho = []
        self.val_rho = []
        self.test_rho = []
        
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
        x1 = F.relu(self.fc1_img(x_img))
        x_img = self.dropout(x1)
        x_img = self.fc2_img(x_img)
        
        return x_img, x1
    
    def rmsi_loss(self, n_outputs, student_rep, teacher_rep, labels):
        T = 0.5
        
        regression_criterion = nn.MSELoss(reduction = 'mean')
        reg_loss = (1.0 - self.alpha) * regression_criterion(n_outputs, labels)
        
        privileged_criterion = nn.CosineEmbeddingLoss(reduction='mean')
        priv_loss = self.alpha * privileged_criterion(student_rep, teacher_rep, torch.ones(labels.shape[0]).to(labels.device.type))
        
        return priv_loss + reg_loss

    
    def train_dataloader(self, s=True):
        return DataLoader(self.train_data, batch_size=batch_sz, num_workers=5, shuffle=s)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=batch_sz, num_workers=5, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0005)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch['images'], batch['label'].reshape((-1))
        n_outputs, _ = self(x)
        n_outputs = n_outputs.squeeze()
        return n_outputs
    
    def training_step(self, batch, batch_idx):
        #print('Training batch: ', batch_idx)
        x_img, y = batch['images'], batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = batch['audio'], batch['EDA'], batch['ECG'], batch['video']
        x_priv_teacher = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, student_rep = self(x_img) # student's predictions
        n_outputs = n_outputs.squeeze()
        _, teacher_rep = self.teacher(x_img, x_priv_teacher) # teacher's predictions 
        loss = self.rmsi_loss(n_outputs, student_rep, teacher_rep, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        return {'loss':loss, 'train_r2':r2, 'train_rho':rho}
    
    def training_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['train_r2'] for x in outputs]).mean()
        print('Training r2: {:.4f}'.format(avg_r2))
        self.train_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['train_rho'] for x in outputs]).mean()
        print('Training rho: {:.4f}'.format(avg_rho))
        self.train_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['loss'] for x in outputs]).mean()
        print('Training mse: {:.4f}'.format(avg_mse))
        self.train_mse.append(avg_mse.item())
    
    def validation_step(self, val_batch, batch_idx):
        x_img, y = val_batch['images'], val_batch['label'].reshape((-1))
        x_audio, x_eda, x_ecg, x_video = val_batch['audio'], val_batch['EDA'], val_batch['ECG'], val_batch['video']
        x_priv_teacher = torch.cat((x_audio, x_eda, x_ecg, x_video), axis = 1)
        n_outputs, student_rep = self(x_img) # student's predictions
        n_outputs = n_outputs.squeeze()
        _, teacher_rep = self.teacher(x_img, x_priv_teacher) # teacher's predictions 
        
        loss = self.rmsi_loss(n_outputs, student_rep, teacher_rep, y)
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        self.log("val_loss", loss)
        self.log("val_r2", r2)
        return {'val_r2':r2, 'val_loss':loss, 'val_rho':rho}
        
    def validation_epoch_end(self, outputs):
        print('========== Epoch: '+str(self.current_epoch)+' ==========')
        avg_r2 = torch.stack([x['val_r2'] for x in outputs]).mean()
        print('Validation r2: {:.4f}'.format(avg_r2))
        self.val_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['val_rho'] for x in outputs]).mean()
        print('Validation rho: {:.4f}'.format(avg_rho))
        self.val_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation mse: {:.4f}'.format(avg_mse))
        self.val_mse.append(avg_mse.item())
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    
    def test_step(self, test_batch, batch_idx):
        x_img, y = test_batch['images'], test_batch['label'].reshape((-1))
        n_outputs, _ = self.forward(x_img)
        n_outputs = n_outputs.squeeze()
        r2 = r2_score(n_outputs, y)
        rho = pearson_corrcoef(n_outputs, y)
        rmsi = nn.MSELoss()
        mse = rmsi(n_outputs, y)
        return {'test_r2':r2, 'test_rho':rho, 'test_mse':mse}
    
    def test_epoch_end(self, outputs):
        avg_r2 = torch.stack([x['test_r2'] for x in outputs]).mean()
        print('Testing r2: {:.4f}'.format(avg_r2))
        self.test_r2.append(avg_r2.item())
        avg_rho = torch.stack([x['test_rho'] for x in outputs]).mean()
        print('Testing rho: {:.4f}'.format(avg_rho))
        self.test_rho.append(avg_rho.item())
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        print('Testing mse: {:.4f}'.format(avg_mse))
        self.test_mse.append(avg_mse.item())
    
    
