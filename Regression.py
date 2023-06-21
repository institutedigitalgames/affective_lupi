#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import mean_squared_error as sk_mse
from scipy.stats import pearsonr
import torch
from torch.utils.data import Subset
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional import r2_score
import RegressionModels as CL
import experiment_variables as ev

# ### Dataset parameters
EPSILON = ev.EPSILON
annotation = ev.annotation
dimension = ev.dimension
WIN_SIZE = ev.WIN_SIZE
STEP = ev.STEP
F_SKIP = ev.F_SKIP
RESCALE_FACTOR = ev.RESCALE_FACTOR

filename = "path to csv"


def TrainCnnImage(dataset, gpu_id):
    GPU_ID = gpu_id
    baseline_r2, baseline_rho, baseline_mse = [], [], []
    
    train_r2, validation_r2, test_r2 = [], [], []
    train_mse, validation_mse, test_mse = [], [], []
    train_rho, validation_rho, test_rho = [], [], []
    train_gt, test_gt, val_gt = [], [], []
    train_preds, test_preds, val_preds = [], [], []
    
    fold = 1
    group_k_fold = GroupKFold(n_splits=5)
    for train_val_index, test_index in group_k_fold.split(ids.reshape((-1,1)), ids, np.asarray(ids, dtype=int)):
        ids_train_val = ids[train_val_index]
        gss = GroupShuffleSplit(n_splits=1, test_size=2, random_state=SEED)
        train_val_data = Subset(dataset, train_val_index)
        for train_index, val_index in gss.split(ids_train_val.reshape((-1,1)), ids_train_val, np.asarray(ids_train_val, dtype=int)):
            print('CnnImage Fold: '+str(fold)+' --> Creating training, validation and testing sets')
        
        train_data = Subset(train_val_data, train_index)
        val_data = Subset(train_val_data, val_index)
        test_data = Subset(dataset, test_index)
        
        temp = data[dimension+annotation+" sample"].iloc[train_index]
        base_predictions = np.ones((temp.shape[0],)) * temp.mean()
        base_r2 = sk_r2(temp, base_predictions)
        base_rho = pearsonr(temp, base_predictions)[0]
        base_mse = sk_mse(temp, base_predictions)
       
        print('Baseline R2: {:3f}'.format(base_r2))
        baseline_r2.append(base_r2)
        print('Baseline Rho: {:3f}'.format(base_rho))
        baseline_rho.append(base_rho)
        print('Baseline MSE: {:3f}'.format(base_mse))
        baseline_mse.append(base_mse)
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.CnnImage(train_data, val_data, test_data)
        r_dir='CnnImage_'+dimension+'_WINSIZE{}_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 50,
                          callbacks = [early_stop, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_r2.append(model.train_r2)
        train_mse.append(model.train_mse)
        train_rho.append(model.train_rho)
        train_predictions = trainer.predict(model, dataloaders=model.train_dataloader(s=False))
        y_train = [data['label'].squeeze() for i, data in enumerate(model.train_dataloader(s=False), 0)]
        train_preds.append(train_predictions)
        train_gt.append(y_train)
        
        validation_r2.append(model.val_r2)
        validation_mse.append(model.val_mse)
        validation_rho.append(model.val_rho)
        val_predictions = trainer.predict(model, dataloaders=model.val_dataloader())
        y_val = [data['label'].squeeze() for i, data in enumerate(model.val_dataloader(), 0)]
        val_preds.append(val_predictions)
        val_gt.append(y_val)
        
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.CnnImage(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        test_predictions = trainer.predict(best_model, dataloaders=best_model.test_dataloader())
        y_test = [data['label'].squeeze() for i, data in enumerate(best_model.test_dataloader(), 0)]
        test_preds.append(test_predictions)
        test_gt.append(y_test)
        
        test_r2.append(best_model.test_r2)
        test_rho.append(best_model.test_rho)
        test_mse.append(best_model.test_mse)
     
        fold += 1
        
    
    preds = [train_preds, val_preds, test_preds]
    gts = [train_gt, val_gt, test_gt]
    fname = r_dir.format(WIN_SIZE)
    with open(fname+'predictions', 'wb') as fp:
        pickle.dump(preds, fp)
        
    with open(fname+'gts', 'wb') as fp:
        pickle.dump(gts, fp)
    
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_r2_CnnImage.npy', np.asarray(train_r2))
    np.save(fname+'valid_r2_CnnImage.npy', np.asarray(validation_r2))
    np.save(fname+'test_r2_CnnImage.npy', np.asarray(test_r2))
    np.save(fname+'base_r2_CnnImage.npy', np.asarray(baseline_r2))
    
    np.save(fname+'train_rho_CnnImage.npy', np.asarray(train_rho))
    np.save(fname+'valid_rho_CnnImage.npy', np.asarray(validation_rho))
    np.save(fname+'test_rho_CnnImage.npy', np.asarray(test_rho))
    np.save(fname+'base_rho_CnnImage.npy', np.asarray(baseline_rho))
    
    np.save(fname+'train_mse_CnnImage.npy', np.asarray(train_mse))
    np.save(fname+'valid_mse_CnnImage.npy', np.asarray(validation_mse))
    np.save(fname+'test_mse_CnnImage.npy', np.asarray(test_mse))
    np.save(fname+'base_mse_CnnImage.npy', np.asarray(baseline_mse))



def TrainFusionModel(dataset, gpu_id):
    GPU_ID = gpu_id
    baseline_r2, baseline_rho, baseline_mse = [], [], []
    
    train_r2, validation_r2, test_r2 = [], [], []
    train_mse, validation_mse, test_mse = [], [], []
    train_rho, validation_rho, test_rho = [], [], []
    train_gt, test_gt, val_gt = [], [], []
    train_preds, test_preds, val_preds = [], [], []
    best_teachers = []
    
    fold = 1
    group_k_fold = GroupKFold(n_splits=5)
    for train_val_index, test_index in group_k_fold.split(ids.reshape((-1,1)), ids, np.asarray(ids, dtype=int)):
        ids_train_val = ids[train_val_index]
        gss = GroupShuffleSplit(n_splits=1, test_size=2, random_state=SEED)
        train_val_data = Subset(dataset, train_val_index)
        for train_index, val_index in gss.split(ids_train_val.reshape((-1,1)), ids_train_val, np.asarray(ids_train_val, dtype=int)):
            print('FusionModel Fold: '+str(fold)+' --> Creating training, validation and testing sets')
        
        train_data = Subset(train_val_data, train_index)
        val_data = Subset(train_val_data, val_index)
        test_data = Subset(dataset, test_index)
        
        temp = data[dimension+annotation+" sample"].iloc[train_index]
        base_predictions = np.ones((temp.shape[0],)) * temp.mean()
        base_r2 = sk_r2(temp, base_predictions)
        base_rho = pearsonr(temp, base_predictions)[0]
        base_mse = sk_mse(temp, base_predictions)
       
        print('Baseline R2: {:3f}'.format(base_r2))
        baseline_r2.append(base_r2)
        print('Baseline Rho: {:3f}'.format(base_rho))
        baseline_rho.append(base_rho)
        print('Baseline MSE: {:3f}'.format(base_mse))
        baseline_mse.append(base_mse)
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.FusionModel(train_data, val_data, test_data)
        r_dir='FusionModel_'+dimension+'_WINSIZE_2_{}_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 150,
                          callbacks = [early_stop, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_r2.append(model.train_r2)
        train_mse.append(model.train_mse)
        train_rho.append(model.train_rho)
        train_predictions = trainer.predict(model, dataloaders=model.train_dataloader(s=False))
        y_train = [data['label'].squeeze() for i, data in enumerate(model.train_dataloader(s=False), 0)]
        train_preds.append(train_predictions)
        train_gt.append(y_train)
        
        validation_r2.append(model.val_r2)
        validation_mse.append(model.val_mse)
        validation_rho.append(model.val_rho)
        val_predictions = trainer.predict(model, dataloaders=model.val_dataloader())
        y_val = [data['label'].squeeze() for i, data in enumerate(model.val_dataloader(), 0)]
        val_preds.append(val_predictions)
        val_gt.append(y_val)
        
        # test accuracy
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.FusionModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        test_predictions = trainer.predict(best_model, dataloaders=best_model.test_dataloader())
        y_test = [data['label'].squeeze() for i, data in enumerate(best_model.test_dataloader(), 0)]
        test_preds.append(test_predictions)
        test_gt.append(y_test)
        
        test_r2.append(best_model.test_r2)
        test_rho.append(best_model.test_rho)
        test_mse.append(best_model.test_mse)
        
        best_teachers.append(checkpoint_callback_loss.best_model_path)
        
        fold += 1

    
    preds = [train_preds, val_preds, test_preds]
    gts = [train_gt, val_gt, test_gt]
    fname = r_dir.format(WIN_SIZE)
    with open(fname+'predictions', 'wb') as fp:
        pickle.dump(preds, fp)
        
    with open(fname+'gts', 'wb') as fp:
        pickle.dump(gts, fp)
        
    with open(fname+'teachers', 'wb') as fp:
        pickle.dump(best_teachers, fp)
    
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_r2_FusionModel.npy', np.asarray(train_r2))
    np.save(fname+'valid_r2_FusionModel.npy', np.asarray(validation_r2))
    np.save(fname+'test_r2_FusionModel.npy', np.asarray(test_r2))
    np.save(fname+'base_r2_FusionModel.npy', np.asarray(baseline_r2))
    
    np.save(fname+'train_rho_FusionModel.npy', np.asarray(train_rho))
    np.save(fname+'valid_rho_FusionModel.npy', np.asarray(validation_rho))
    np.save(fname+'test_rho_FusionModel.npy', np.asarray(test_rho))
    np.save(fname+'base_rho_FusionModel.npy', np.asarray(baseline_rho))
    
    np.save(fname+'train_mse_FusionModel.npy', np.asarray(train_mse))
    np.save(fname+'valid_mse_FusionModel.npy', np.asarray(validation_mse))
    np.save(fname+'test_mse_FusionModel.npy', np.asarray(test_mse))
    np.save(fname+'base_mse_FusionModel.npy', np.asarray(baseline_mse))
    



def TrainTeacherModel(dataset, gpu_id):
    GPU_ID = gpu_id
    baseline_r2, baseline_rho, baseline_mse = [], [], []
    
    train_r2, validation_r2, test_r2 = [], [], []
    train_mse, validation_mse, test_mse = [], [], []
    train_rho, validation_rho, test_rho = [], [], []
    train_gt, test_gt, val_gt = [], [], []
    train_preds, test_preds, val_preds = [], [], []
    best_teachers = []
    
    fold = 1
    group_k_fold = GroupKFold(n_splits=5)
    for train_val_index, test_index in group_k_fold.split(ids.reshape((-1,1)), ids, np.asarray(ids, dtype=int)):
        ids_train_val = ids[train_val_index]
        gss = GroupShuffleSplit(n_splits=1, test_size=2, random_state=SEED)
        train_val_data = Subset(dataset, train_val_index)
        for train_index, val_index in gss.split(ids_train_val.reshape((-1,1)), ids_train_val, np.asarray(ids_train_val, dtype=int)):
            print('TeacherModel Fold: '+str(fold)+' --> Creating training, validation and testing sets')
        
        train_data = Subset(train_val_data, train_index)
        val_data = Subset(train_val_data, val_index)
        test_data = Subset(dataset, test_index)
        
        temp = data[dimension+annotation+" sample"].iloc[train_index]
        base_predictions = np.ones((temp.shape[0],)) * temp.mean()
        base_r2 = sk_r2(temp, base_predictions)
        base_rho = pearsonr(temp, base_predictions)[0]
        base_mse = sk_mse(temp, base_predictions)
       
        print('Baseline R2: {:3f}'.format(base_r2))
        baseline_r2.append(base_r2)
        print('Baseline Rho: {:3f}'.format(base_rho))
        baseline_rho.append(base_rho)
        print('Baseline MSE: {:3f}'.format(base_mse))
        baseline_mse.append(base_mse)
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.TeacherModel(train_data, val_data, test_data)
        r_dir='TeacherModel_'+dimension+'_WINSIZE{}_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 150,
                          callbacks = [early_stop, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_r2.append(model.train_r2)
        train_mse.append(model.train_mse)
        train_rho.append(model.train_rho)
        train_predictions = trainer.predict(model, dataloaders=model.train_dataloader(s=False))
        y_train = [data['label'].squeeze() for i, data in enumerate(model.train_dataloader(s=False), 0)]
        train_preds.append(train_predictions)
        train_gt.append(y_train)
        
        validation_r2.append(model.val_r2)
        validation_mse.append(model.val_mse)
        validation_rho.append(model.val_rho)
        val_predictions = trainer.predict(model, dataloaders=model.val_dataloader())
        y_val = [data['label'].squeeze() for i, data in enumerate(model.val_dataloader(), 0)]
        val_preds.append(val_predictions)
        val_gt.append(y_val)
        
        # test accuracy
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.TeacherModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        test_predictions = trainer.predict(best_model, dataloaders=best_model.test_dataloader())
        y_test = [data['label'].squeeze() for i, data in enumerate(best_model.test_dataloader(), 0)]
        test_preds.append(test_predictions)
        test_gt.append(y_test)
        
        test_r2.append(best_model.test_r2)
        test_rho.append(best_model.test_rho)
        test_mse.append(best_model.test_mse)
        
        best_teachers.append(checkpoint_callback_loss.best_model_path)
        
        fold += 1

        
    preds = [train_preds, val_preds, test_preds]
    gts = [train_gt, val_gt, test_gt]
    fname = r_dir.format(WIN_SIZE)
    with open(fname+'predictions', 'wb') as fp:
        pickle.dump(preds, fp)
        
    with open(fname+'gts', 'wb') as fp:
        pickle.dump(gts, fp)
        
    with open(fname+'teachers', 'wb') as fp:
        pickle.dump(best_teachers, fp)
    
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_r2_TeacherModel.npy', np.asarray(train_r2))
    np.save(fname+'valid_r2_TeacherModel.npy', np.asarray(validation_r2))
    np.save(fname+'test_r2_TeacherModel.npy', np.asarray(test_r2))
    np.save(fname+'base_r2_TeacherModel.npy', np.asarray(baseline_r2))
    
    np.save(fname+'train_rho_TeacherModel.npy', np.asarray(train_rho))
    np.save(fname+'valid_rho_TeacherModel.npy', np.asarray(validation_rho))
    np.save(fname+'test_rho_TeacherModel.npy', np.asarray(test_rho))
    np.save(fname+'base_rho_TeacherModel.npy', np.asarray(baseline_rho))
    
    np.save(fname+'train_mse_TeacherModel.npy', np.asarray(train_mse))
    np.save(fname+'valid_mse_TeacherModel.npy', np.asarray(validation_mse))
    np.save(fname+'test_mse_TeacherModel.npy', np.asarray(test_mse))
    np.save(fname+'base_mse_TeacherModel.npy', np.asarray(baseline_mse))



def TrainStudentModel(dataset, teachers_p, alpha, gpu_id, fn):
    GPU_ID = gpu_id
    baseline_r2, baseline_rho, baseline_mse = [], [], []
    
    train_r2, validation_r2, test_r2 = [], [], []
    train_mse, validation_mse, test_mse = [], [], []
    train_rho, validation_rho, test_rho = [], [], []
    train_gt, test_gt, val_gt = [], [], []
    train_preds, test_preds, val_preds = [], [], []
        
    fold = 1
    group_k_fold = GroupKFold(n_splits=5)
    for train_val_index, test_index in group_k_fold.split(ids.reshape((-1,1)), ids, np.asarray(ids, dtype=int)):
        ids_train_val = ids[train_val_index]
        gss = GroupShuffleSplit(n_splits=1, test_size=2, random_state=SEED)
        train_val_data=Subset(dataset, train_val_index)
        for train_index, val_index in gss.split(ids_train_val.reshape((-1,1)), ids_train_val, np.asarray(ids_train_val, dtype=int)):
            print('StudentModel Fold: '+str(fold)+' --> Creating training, validation and testing sets')
        
        train_data = Subset(train_val_data, train_index)
        val_data = Subset(train_val_data, val_index)
        test_data = Subset(dataset, test_index)
        
        temp = data[dimension+annotation+" sample"].iloc[train_index]
        base_predictions = np.ones((temp.shape[0],)) * temp.mean()
        base_r2 = sk_r2(temp, base_predictions)
        base_rho = pearsonr(temp, base_predictions)[0]
        base_mse = sk_mse(temp, base_predictions)
       
        print('Baseline R2: {:3f}'.format(base_r2))
        baseline_r2.append(base_r2)
        print('Baseline Rho: {:3f}'.format(base_rho))
        baseline_rho.append(base_rho)
        print('Baseline MSE: {:3f}'.format(base_mse))
        baseline_mse.append(base_mse)
        if 'Fusion' in fn:
            teacher = CL.FusionModel(train_data, val_data, test_data)
            r_dir = 'StudentModel_'+dimension+'_WINSIZE_fusion_{}_{}_'+str(SEED)+'/'
        elif 'Teacher' in fn:
            teacher = CL.TeacherModel(train_data, val_data, test_data)
            r_dir = 'StudentModel_'+dimension+'_WINSIZE_teacher_{}_{}_'+str(SEED)+'/'
        teacher_state = torch.load(teachers_p[fold-1], map_location='cuda:1')
        teacher.load_state_dict(teacher_state['state_dict'])
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.StudentModel(train_data, val_data, test_data, teacher, alpha)
        
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 150,
                          callbacks = [early_stop, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE, alpha))
        
        train_res = trainer.fit(model)
        train_r2.append(model.train_r2)
        train_mse.append(model.train_mse)
        train_rho.append(model.train_rho)
        train_predictions = trainer.predict(model, dataloaders=model.train_dataloader(s=False))
        y_train = [data['label'].squeeze() for i, data in enumerate(model.train_dataloader(s=False), 0)]
        train_preds.append(train_predictions)
        train_gt.append(y_train)
        
        validation_r2.append(model.val_r2)
        validation_mse.append(model.val_mse)
        validation_rho.append(model.val_rho)
        val_predictions = trainer.predict(model, dataloaders=model.val_dataloader())
        y_val = [data['label'].squeeze() for i, data in enumerate(model.val_dataloader(), 0)]
        val_preds.append(val_predictions)
        val_gt.append(y_val)
        
        # test accuracy
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.StudentModel(train_data, val_data, test_data, teacher, alpha)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        test_predictions = trainer.predict(best_model, dataloaders=best_model.test_dataloader())
        y_test = [data['label'].squeeze() for i, data in enumerate(best_model.test_dataloader(), 0)]
        test_preds.append(test_predictions)
        test_gt.append(y_test)
        
        test_r2.append(best_model.test_r2)
        test_rho.append(best_model.test_rho)
        test_mse.append(best_model.test_mse)
        
        fold += 1
        
        
        
    preds = [train_preds, val_preds, test_preds]
    gts = [train_gt, val_gt, test_gt]
    fname = r_dir.format(WIN_SIZE, alpha)
    with open(fname+'predictions', 'wb') as fp:
        pickle.dump(preds, fp)
        
    with open(fname+'gts', 'wb') as fp:
        pickle.dump(gts, fp)
        
    
    fname = r_dir.format(WIN_SIZE, alpha)
    np.save(fname+'train_r2_StudentModel.npy', np.asarray(train_r2))
    np.save(fname+'valid_r2_StudentModel.npy', np.asarray(validation_r2))
    np.save(fname+'test_r2_StudentModel.npy', np.asarray(test_r2))
    np.save(fname+'base_r2_StudentModel.npy', np.asarray(baseline_r2))
    
    np.save(fname+'train_rho_StudentModel.npy', np.asarray(train_rho))
    np.save(fname+'valid_rho_StudentModel.npy', np.asarray(validation_rho))
    np.save(fname+'test_rho_StudentModel.npy', np.asarray(test_rho))
    np.save(fname+'base_rho_StudentModel.npy', np.asarray(baseline_rho))
    
    np.save(fname+'train_mse_StudentModel.npy', np.asarray(train_mse))
    np.save(fname+'valid_mse_StudentModel.npy', np.asarray(validation_mse))
    np.save(fname+'test_mse_StudentModel.npy', np.asarray(test_mse))
    np.save(fname+'base_mse_StudentModel.npy', np.asarray(baseline_mse))

if __name__ == '__main__':
    data, ids = CL.regression_data(filename)
    data['time']=data['time']//150+1
    dataset = CL.RegressionDataset(data, transform=transforms.Compose([CL.ToTensor()]))
    gpi =  0 if dimension=='arousal' else 1
    for SEED in [49]:
    
            seed_everything(SEED)
            #train fusion teacher
            TrainFusionModel(dataset, [gpi]) 
            #train model without LUPI
            TrainCnnImage(dataset, [gpi])
            #train privileged teacher
            TrainTeacherModel(dataset, [gpi])
            #load fusion teacher
            r_dir = 'FusionModel_'+dimension+'_WINSIZE_2_{}_'+str(SEED)+'/'
            fname = r_dir.format(WIN_SIZE)
            with open(fname+'teachers', 'rb') as fp:
                teachers_p = pickle.load(fp)
            #train student model with a=0.25 and fusion teacher    
            TrainStudentModel(dataset, teachers_p, 0.25, [gpi],fn = fname)
            #load privileged teacher
            r_dir= 'TeacherModel_'+dimension+'_WINSIZE{}_'+str(SEED)+'/'
            fname = r_dir.format(WIN_SIZE)
            with open(fname+'teachers', 'rb') as fp:
                teachers_p = pickle.load(fp)
            #train student model with a=0.25 and privileged teacher                    
            TrainStudentModel(dataset, teachers_p, 0.25, [1],fn = fname)
          