from __future__ import print_function
import os
import time
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
import random
import pandas as pd
import yaml
import csv
import pickle


class Processor():
    """
        Processor for Pose-based training and testing
    """

    def __init__(self, arg):

        self.arg = arg
        self.model_saved_dir = os.path.join(
            self.arg.model_saved_dir, arg.Experiment_name)
        self.work_dir = os.path.join(
            self.arg.work_dir, arg.Experiment_name)
        self.save_arg()

        if arg.phase == 'train':
            if os.path.exists(self.model_saved_dir):
                print('log_dir: ', self.model_saved_dir, 'already exist')
            else:
                os.makedirs(self.model_saved_dir, exist_ok=True)

            if os.path.exists(self.work_dir):
                print('log_dir: ', self.work_dir, 'already exist')
            else:
                os.makedirs(self.work_dir, exist_ok=True)

            self.training_info_file = os.path.join(
                self.work_dir, 'epoch_info_training.csv')
            self.print_log('Save epoch results to {}'.format(
                self.training_info_file), print_time=False)

        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.init_environment()
        self.lr = self.arg.base_lr

    def init_environment(self):
        self.result = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=self.arg.start_epoch, best_acc=0)
        self.epoch_training_info = pd.DataFrame(
            columns=['Epoch', 'Train loss', 'Train accuracy', 'Val loss', 'Val accuracy'])

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        with open('{}/config.yaml'.format(self.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):

        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)

        if self.arg.weights:
            weights = torch.load(self.arg.weights)
            try:
                self.model.load_state_dict(weights)
                self.print_log(
                    'Load weights from {}.'.format(self.arg.weights))
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            self.print_log("Cannot load optimizer, try SGD or Adam")
            raise ValueError()

        # Set up the CosineAnnealing LR scheduler
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.arg.t_max)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self):

        self.model.train()
        loader = self.data_loader['train']
        loss_value = []
        result_frag = []
        label_frag = []

        for (data, label) in tqdm(loader):

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            loss_value.append(loss.data.cpu().numpy())
            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        self.lr_scheduler.step()

        self.epoch_info['train_loss'] = np.mean(loss_value)
        self.label = np.concatenate(label_frag)
        self.result = np.concatenate(result_frag)

        # show top-k accuracy
        for k in self.arg.show_topk:
            accuracy = self.show_topk(k)
            self.epoch_info['train_acc'] = accuracy

    def validation(self):
        self.model.eval()
        loader = self.data_loader['val']
        loss_value = []
        result_frag = []
        label_frag = []

        for (data, label) in tqdm(loader):

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        self.epoch_info['val_loss'] = np.mean(loss_value)

        # show top-k accuracy
        for k in self.arg.show_topk:
            accuracy = self.show_topk(k)
            self.epoch_info['val_acc'] = accuracy
            # self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy),print_time=False)
        return accuracy

    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for (data, label) in tqdm(loader):

            # get data
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        self.epoch_info['test_loss'] = np.mean(loss_value)

        # show top-k accuracy
        for k in [1, 5]:
            accuracy = self.show_topk(k)
            self.epoch_info[f'top{k}_acc'] = accuracy
        return accuracy

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        return accuracy

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.epoch_info['epoch'] = epoch

                # training
                self.print_log('Training epoch: {}'.format(epoch))
                self.train()

                # Evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or \
                        (epoch + 1 == self.arg.num_epoch):
                    accuracy = self.validation()

                # Print epoch
                self.print_log('Statistic epoch {}'.format(epoch))
                self.show_epoch_info()

                if self.meta_info['best_acc'] < accuracy:
                    self.meta_info['best_acc'] = accuracy
                    # Save checkpoint best_acc
                    filename = 'best_acc.pt'
                    torch.save(self.model.state_dict(), os.path.join(
                        self.model_saved_dir, filename))
                    self.print_log('Checkpoint best accuracy in epoch {}'.format(
                        epoch), print_time=False)

                # Save model
                if ((epoch + 1) % self.arg.save_interval == 0) or \
                        (epoch + 1 == self.arg.num_epoch):
                    filename = self.arg.Experiment_name + \
                        '-' + str(epoch) + '.pt'
                    torch.save(self.model.state_dict(), os.path.join(
                        self.model_saved_dir, filename))
                    self.print_log('Save model to {}'.format(
                        filename), print_time=False)

                # Save result
                if not os.path.exists(self.training_info_file):
                    with open(self.training_info_file, 'w') as f:
                        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

                with open(self.training_info_file, 'a', newline='') as f:
                    row = self.epoch_info
                    row = [row['epoch'], row['train_loss'], row['train_acc'],
                           row['val_loss'], row['val_acc']]
                    writer = csv.writer(f)
                    writer.writerow(row)

        elif self.arg.phase == 'test':

            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.print_log('Test Start:')
            self.test()
            self.print_log('Done.\n')

            self.print_log('Test Loss: {}\t Top 1 Accuracy:{} \t Top 5 Accuracy: {}'
                           .format(self.epoch_info['test_loss'], self.epoch_info['top1_acc'],
                                   self.epoch_info['top5_acc']))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.print_log('\t{}: {:.4f}'.format(k, v), print_time=False)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
