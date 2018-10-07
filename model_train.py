#coding=utf-8
from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from model_ResnetX_TextCNN.ResNetTextCNN import *
from model_ResnetX_TextCNN.data_utils import DataLoader
import time
import os
import copy
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

# Training Settings
parser.add_argument('--mode',default='train',type=str,help="train or test ? ")
parser.add_argument('--cnn_model', default='resnet101',type=str,help="please choose a kind of model")
parser.add_argument('--GPU_no',default=4,type=int,help="please choose a GPU No.")

# Training Data Settings
parser.add_argument('--images_dir', default="../data/images_buckets/",type=str,help="path of images")
parser.add_argument('--first_category',default="42",type=str,help="select one first_category to train ")
parser.add_argument('--texts_dir',default="../data/text_info_new/",type=str,help="path of images text info ")

# Parameters Settings
parser.add_argument('--num_epochs',default=25,type=int)
parser.add_argument('--num_class',default=5,type=int)
parser.add_argument('--topN_acc',default=2,type=int)
parser.add_argument('--batch_size',default=2,type=int,help="Iteration batch size")
parser.add_argument('--num_workers',default=1,type=int,help="parallel threads num")

# Testing Settings
parser.add_argument('--test_image',default="",type=str,help="image path is going to be predicted")

args = parser.parse_args()


class ModelTrainer(object):
    def __init__(self):

        # Settings
        self.mode = args.mode               # train or test
        self.cnn_model = args.cnn_model     # resnet101
        self.GPU_no = args.GPU_no

        self.images_dir = args.images_dir
        self.first_category = args.first_category
        self.texts_dir = args.texts_dir

        self.num_epochs = args.num_epochs
        self.num_class = args.num_class
        self.topN_acc = args.topN_acc
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # data preparation
        self.data_paras = {
            "images_dir": self.images_dir,
            "first_category": self.first_category,
            "texts_dir": self.texts_dir,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size
        }
        self.dataLoader = DataLoader(self.data_paras)
        self.device = torch.device("cuda:{}".format(self.GPU_no) if torch.cuda.is_available() else "cpu")
        self._init_model()

    def _init_model(self):
        model_args = {
        'cnn':{
            'num_class':self.num_class,
            'resnet_version':101,
            },
        'text_cnn':{
            'vocb_size': self.dataLoader.vocb_size,
            'max_len': self.dataLoader.max_len,
            'n_class': self.dataLoader.n_class,
            'dim': self.dataLoader.word_dim,
            'embedding_matrix': self.dataLoader.embedding_matrix,
            },
        'device':self.device,
        }
        self.model = ResnetXTextCNN(args=model_args)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        self.dataset_sizes = {x: len(self.dataLoader.image_datasets[x]) for x in ['train', 'val']}
        self.class_sizes = {
            x: [len(os.listdir(os.path.join(self.images_dir,self.first_category,x, str(i)))) for i in range(self.num_class)]
            for x in ['train', 'val']}

        self.class_names = self.dataLoader.image_datasets['train'].classes

    def computeTopNCorrects(self,topN_running_corrects,outputs,labels):
        """
        计算topN acc
        :param topN_running_corrects:
        :param outputs:
        :param labels:
        :return:  correct num
        """
        outputs_numpy = outputs.cpu().data.numpy()
        labels_numpy = labels.cpu().data.numpy()
        for preds, label in zip(outputs_numpy, labels_numpy):
            preds = sorted([(i,val) for i,val in enumerate(preds)],key=lambda x:x[1],reverse=True)
            preds_labels = [val[0] for val in preds[:self.topN_acc]]
            if label in preds_labels: topN_running_corrects += 1
        return topN_running_corrects


    def train_model(self):
        """
        预测类别 （bucket）
        :return:
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []

        for epoch in range(self.num_epochs):
            i_epoch_since = time.time()
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.exp_lr_scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_corrects_class = [0 for _ in range(self.num_class)]

                topN_running_corrects = 0

                # Iterate over data.
                for inputs, labels_path in self.dataLoader.dataloaders[phase]:
                    inputs_img = inputs.to(self.device)
                    labels = labels_path[0].to(self.device)
                    filespath = labels_path[1]
                    inputs_text = self.dataLoader.load_texts_batch(filespath) # word_with_id
                    inputs_text = Variable(torch.LongTensor(inputs_text))
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs_img, inputs_text)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    labels = labels.long()
                    running_corrects += torch.sum(preds == labels.data)

                    running_corrects_class = [val+torch.sum(labels.data[labels.data == preds] == i)
                                              for i,val in enumerate(running_corrects_class)]
                    topN_running_corrects = self.computeTopNCorrects(topN_running_corrects,outputs,labels)

                epoch_loss = running_loss / self.dataset_sizes[phase]

                # top1 acc
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                epoch_acc_class = [rc.double() / samples_num for rc,samples_num in
                                   zip(running_corrects_class, self.class_sizes[phase])]
                # topN acc
                topN_epoch_acc = topN_running_corrects / self.dataset_sizes[phase]

                if phase == "train":
                    loss_train.append(epoch_loss)
                    acc_train.append(epoch_acc)
                elif phase == 'val':
                    loss_val.append(epoch_loss)
                    acc_val.append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}  top2Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc,topN_epoch_acc))
                for i,acc in enumerate(epoch_acc_class):
                    print("     {} class: Acc: {:.4f}".format(i,acc))
                # deep copy the model
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            i_time_elapsed = time.time() - i_epoch_since
            print('    ecpoch:{} completed in {:.0f}m {:.0f}s'.format(epoch, i_time_elapsed // 60,
                                                                      i_time_elapsed % 60))
        with open("./training_loss_acc_class.csv",'w') as file:
            file.write("{},{},{},{}\n".format("train_loss","val_loss","train_acc","val_acc"))
            file.writelines(["{},{},{},{}\n".format(loss_train[i],loss_val[i],acc_train[i],acc_val[i])
                             for i in range(len(loss_train))])

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # save all the parameters
        torch.save(self.model.state_dict(), 'params_class.pkl')

        return self.model
    def predict_class(self):
        """
        预测类别（bucket）
        :return:
        """
        self.model.load_state_dict(torch.load("params.pkl"))
        image = Image.open(args.image_path)
        image_tensor = self.dataLoader.data_transforms['val'](image).float()
        image_tensor.unsqueeze_(0)
        input = Variable(image_tensor).to(self.device)
        outputs = self.model(input)
        _, prediction_tensor = torch.max(outputs.data, 1)
        prediction = prediction_tensor.cpu().numpy()[0]
        return prediction

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_model()