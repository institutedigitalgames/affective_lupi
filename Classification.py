#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from torch.utils.data import  Subset
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import ClassificationModels as CL
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

### Running parameters
def TrainCnnImage(dataset, gpu_id):
    GPU_ID = gpu_id
    train_accuracies, validation_accuracies, test_accuracies, baseline, mode = [], [], [], [], []
    baseline_maj=[]
    fold = 1
    group_k_fold = GroupKFold(n_splits=5,)
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
        temp_test =data[dimension+annotation+" sample"].iloc[test_index]
        
        temp[temp >= 0] =1
        temp[temp < 0] = 0
        temp_test[temp_test >= 0] =1
        temp_test[temp_test < 0] = 0
        
        
        base_maj = temp.sum() / temp.shape[0]
        if base_maj < 0.5:
            base_maj = 1 - base_maj
            mclass=0
        else:
            mclass=1
        base = (temp_test==mclass).astype(int)
        base=base.sum() / base.shape[0]
        print('Baseline: {:3f}'.format(base))
        baseline.append(base)
        baseline_maj.append(base_maj)
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_acc = ModelCheckpoint(monitor = 'val_acc',
                                                  mode = 'max',
                                                  verbose = True)
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.CnnImage(train_data, val_data, test_data,)
        r_dir='CnnImage_class_'+dimension+'_WINSIZE{}_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 50,
                          callbacks = [early_stop, checkpoint_callback_acc, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_accuracies.append(model.train_acc)
        validation_accuracies.append(model.val_acc)
        
        # test accuracy
        print(checkpoint_callback_acc.best_model_path)
        
        best_model = CL.CnnImage(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_acc.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        temp_acc = best_model.test_acc
        
        # test loss
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.CnnImage(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        #save best accuracy
        if temp_acc < best_model.test_acc:
            test_accuracies.append(best_model.test_acc)
            mode.append(1) # loss
        else:
            test_accuracies.append(temp_acc)
            mode.append(0) # accuracy
        
        fold += 1
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_acc_CnnImage.npy', np.asarray(train_accuracies))
    np.save(fname+'valid_acc_CnnImage.npy', np.asarray(validation_accuracies))
    np.save(fname+'test_acc_CnnImage.npy', np.asarray(test_accuracies))
    np.save(fname+'base_acc_CnnImage.npy', np.asarray(baseline))
    np.save(fname+'base_maj_acc_CnnImage.npy', np.asarray(baseline_maj))
    np.save(fname+'mode_CnnImage.npy', np.asarray(mode))
    return train_accuracies, validation_accuracies, test_accuracies, baseline, mode



def TrainFusionModel(dataset, gpu_id):
    GPU_ID = gpu_id
    train_accuracies, validation_accuracies, test_accuracies, baseline, mode = [], [], [], [], []
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
        temp[temp >= 0] =1
        temp[temp < 0] = 0
        base = temp.sum() / temp.shape[0]
        if base < 0.5:
            base = 1 - base
        print('Baseline: {:3f}'.format(base))
        baseline.append(base)
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_acc = ModelCheckpoint(monitor = 'val_acc',
                                                  mode = 'max',
                                                  verbose = True)
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.FusionModel(train_data, val_data, test_data)
        r_dir='FusionModel_class_'+dimension+'_WINSIZE_2_{}'+'_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 50,
                          callbacks = [early_stop, checkpoint_callback_acc, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_accuracies.append(model.train_acc)
        validation_accuracies.append(model.val_acc)
        
        # test accuracy
        print(checkpoint_callback_acc.best_model_path)
        
        best_model = CL.FusionModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_acc.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        temp_acc = best_model.test_acc
        
        # test loss
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.FusionModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        #save best accuracy
        if temp_acc < best_model.test_acc:
            test_accuracies.append(best_model.test_acc)
            best_teachers.append(checkpoint_callback_loss.best_model_path)
            mode.append(1) # loss
        else:
            test_accuracies.append(temp_acc)
            best_teachers.append(checkpoint_callback_acc.best_model_path)
            mode.append(0) # accuracy
        
        fold += 1
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_acc_FusionModel.npy', np.asarray(train_accuracies))
    np.save(fname+'valid_acc_FusionModel.npy', np.asarray(validation_accuracies))
    np.save(fname+'test_acc_FusionModel.npy', np.asarray(test_accuracies))
    np.save(fname+'base_acc_FusionModel.npy', np.asarray(baseline))
    np.save(fname+'mode_FusionModel.npy', np.asarray(mode))
    return train_accuracies, validation_accuracies, test_accuracies, baseline, mode, best_teachers



def TrainTeacherModel(dataset, gpu_id):
    GPU_ID = gpu_id
    train_accuracies, validation_accuracies, test_accuracies, baseline, mode = [], [], [], [], []
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
        temp[temp >= 0] =1
        temp[temp < 0] = 0
        base = temp.sum() / temp.shape[0]
        if base < 0.5:
            base = 1 - base
        print('Baseline: {:3f}'.format(base))
        baseline.append(base)
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_acc = ModelCheckpoint(monitor = 'val_acc',
                                                  mode = 'max',
                                                  verbose = True)
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.TeacherModel(train_data, val_data, test_data)
        r_dir='TeacherModel_class_'+dimension+'_WINSIZE{}'+'_'+str(SEED)+'/'
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 50,
                          callbacks = [early_stop, checkpoint_callback_acc, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE))
        
        train_res = trainer.fit(model)
        train_accuracies.append(model.train_acc)
        validation_accuracies.append(model.val_acc)
        
        # test accuracy
        print(checkpoint_callback_acc.best_model_path)
        
        best_model = CL.TeacherModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_acc.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        temp_acc = best_model.test_acc
        
        # test loss
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.TeacherModel(train_data, val_data, test_data)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        #save best accuracy
        if temp_acc < best_model.test_acc:
            test_accuracies.append(best_model.test_acc)
            mode.append(1) # loss
            best_teachers.append(checkpoint_callback_loss.best_model_path)
        else:
            test_accuracies.append(temp_acc)
            mode.append(0) # accuracy
            best_teachers.append(checkpoint_callback_acc.best_model_path)
            
        
        fold += 1
    fname = r_dir.format(WIN_SIZE)
    np.save(fname+'train_acc_TeacherModel.npy', np.asarray(train_accuracies))
    np.save(fname+'valid_acc_TeacherModel.npy', np.asarray(validation_accuracies))
    np.save(fname+'test_acc_TeacherModel.npy', np.asarray(test_accuracies))
    np.save(fname+'base_acc_TeacherModel.npy', np.asarray(baseline))
    np.save(fname+'mode_TeacherModel.npy', np.asarray(mode))
    return train_accuracies, validation_accuracies, test_accuracies, baseline, mode, best_teachers



def TrainStudentModel(dataset, teachers_p, alpha, gpu_id,fn):
    GPU_ID = gpu_id
    train_accuracies, validation_accuracies, test_accuracies, baseline, mode = [], [], [], [], []
        
    fold = 1
    group_k_fold = GroupKFold(n_splits=5)
    
    for train_val_index, test_index in group_k_fold.split(ids.reshape((-1,1)), ids, np.asarray(ids, dtype=int)):
        ids_train_val = ids[train_val_index]
        gss = GroupShuffleSplit(n_splits=1, test_size=2, random_state=SEED)
        train_val_data = Subset(dataset, train_val_index)
        for train_index, val_index in gss.split(ids_train_val.reshape((-1,1)), ids_train_val, np.asarray(ids_train_val, dtype=int)):
            print('StudentModel Fold: '+str(fold)+' --> Creating training, validation and testing sets')
        
        train_data = Subset(train_val_data, train_index)
        val_data = Subset(train_val_data, val_index)
        test_data = Subset(dataset, test_index)
        
        temp = data[dimension+annotation+" sample"].iloc[train_index]
        temp[temp >= 0] =1
        temp[temp < 0] = 0
        base = temp.sum() / temp.shape[0]
        if base < 0.5:
            base = 1 - base
        print('Baseline: {:3f}'.format(base))
        baseline.append(base)
        
        if 'Fusion' in fn:
            teacher = CL.FusionModel(train_data, val_data, test_data)
            r_dir = 'StudentModel_class_'+dimension+'_WINSIZE_fusion_{}_{}'+'_'+str(SEED)+'/'
        else:
            teacher=CL.TeacherModel(train_data, val_data, test_data)
            r_dir = 'StudentModel_class_'+dimension+'_WINSIZE_teacher_{}_{}'+'_'+str(SEED)+'/'
        teacher_state = torch.load(teachers_p[fold-1])
        teacher.load_state_dict(teacher_state['state_dict'])
        
        early_stop = EarlyStopping(monitor = 'val_loss',
                                    patience = 5,
                                    strict = True,
                                    verbose = True,
                                    mode = 'min')
        
        checkpoint_callback_acc = ModelCheckpoint(monitor = 'val_acc',
                                                  mode = 'max',
                                                  verbose = True)
        
        checkpoint_callback_loss = ModelCheckpoint(monitor = 'val_loss',
                                                   mode = 'min',
                                                   verbose = True)
        
        model = CL.StudentModel(train_data, val_data, test_data, teacher, alpha)
        
        trainer = Trainer(gpus = GPU_ID,
                          max_epochs = 50,
                          callbacks = [early_stop, checkpoint_callback_acc, checkpoint_callback_loss],
                          progress_bar_refresh_rate = 0,
                          default_root_dir = r_dir.format(WIN_SIZE, alpha))
        
        train_res = trainer.fit(model) 
        train_accuracies.append(model.train_acc)
        validation_accuracies.append(model.val_acc)
        
        # test accuracy
        print(checkpoint_callback_acc.best_model_path)
        
        best_model = CL.StudentModel(train_data, val_data, test_data, teacher, alpha)
        best_state = torch.load(checkpoint_callback_acc.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        temp_acc = best_model.test_acc
        
        # test loss
        print(checkpoint_callback_loss.best_model_path)
        
        best_model = CL.StudentModel(train_data, val_data, test_data, teacher, alpha)
        best_state = torch.load(checkpoint_callback_loss.best_model_path)
        best_model.load_state_dict(best_state['state_dict'])
        
        test_res = trainer.test(best_model)
        
        if temp_acc < best_model.test_acc:
            test_accuracies.append(best_model.test_acc)
            mode.append(1) # loss
        else:
            test_accuracies.append(temp_acc)
            mode.append(0) # accuracy
            
        
        fold += 1
        
    fname = r_dir.format(WIN_SIZE, alpha)
    np.save(fname+'train_acc_StudentModel.npy', np.asarray(train_accuracies))
    np.save(fname+'valid_acc_StudentModel.npy', np.asarray(validation_accuracies))
    np.save(fname+'test_acc_StudentModel.npy', np.asarray(test_accuracies))
    np.save(fname+'base_acc_StudentModel.npy', np.asarray(baseline))
    np.save(fname+'mode_StudentModel.npy', np.asarray(mode))
    return train_accuracies, validation_accuracies, test_accuracies, baseline, mode

if __name__ == '__main__':
    for SEED in [49]:
        seed_everything(SEED)
        
        data, ids = CL.classification_data(filename)
        gpi =3 if dimension == 'arousal' else 2 #set gpu
        dataset = CL.ClassificationDataset(data, transform=transforms.Compose([CL.ToTensor()]))


        #train model without LUPI
        _, _, _, _, _ = TrainCnnImage(dataset, [gpi])        
        #train fusion teacher model
        _, _, _, _, _ , best_teachers_fusion = TrainFusionModel(dataset, [gpi]) 
        #save fusion teachers
        r_dir= 'FusionModel_class_'+dimension+'_WINSIZE_2_{}'+'_'+str(SEED)+'/'
        fname =r_dir.format(WIN_SIZE)
        with open(fname+'best_teachers_fusion', 'wb') as fp:
            pickle.dump(best_teachers_fusion, fp)
        #train privileged teachers    
        _, _, _, _, _ , best_teachers = TrainTeacherModel(dataset, [1])
        #save privileged teachers
        r_dir='TeacherModel_class_'+dimension+'_WINSIZE{}'+'_'+str(SEED)+'/'
        fname = r_dir.format(WIN_SIZE)
        with open(fname+'best_teachers', 'wb') as fp:
            pickle.dump(best_teachers, fp)
            
        #load privileged teachers
        r_dir='TeacherModel_class_'+dimension+'_WINSIZE{}'+'_'+str(SEED)+'/'
        fname = r_dir.format(WIN_SIZE)
        with open(fname+'best_teachers', 'rb') as fp:
            teachers_p = pickle.load(fp)
        #train student with a=0.25 and privileged teacher
        train_accuracies, validation_accuracies, test_accuracies, baseline, mode = TrainStudentModel(dataset, teachers_p, 0.25, [1],fn=fname)
        
        #load fusion teacher
        r_dir='FusionModel_class_'+dimension+'_WINSIZE_2_{}'+'_'+str(SEED)+'/'
        fname = r_dir.format(WIN_SIZE)
        with open(fname+'best_teachers_fusion', 'rb') as fp:
            teachers_p = pickle.load(fp)
        #train student with a=0.25 and fusion teacher
        train_accuracies, validation_accuracies, test_accuracies, baseline, mode = TrainStudentModel(dataset, teachers_p, 0.25, [gpi],fn=fname)
        
        
    
    
    
    
    
    
    
