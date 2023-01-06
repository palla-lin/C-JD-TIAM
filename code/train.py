# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/05 
"""
import torch
import argparse
from loader import Vocab, Processor
from model import Netv5, Netv6, Netv7, Netv8, Netv9
from MyTrainer import *
# from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='proj')
parser.add_argument('-M', '--model', type=str)
parser.add_argument('-ds', '--data_size', type=int)
parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-model_name', '--model_name', type=str)
parser.add_argument('-wandb', '--wandb', type=str)
parser.add_argument('-output_dir', '--output_dir', type=str)
parser.add_argument('-num_layers', '--num_layers', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-dr', '--dropout_rate', type=float)
parser.add_argument('-kernel', '--kernel_wins', type=str)
parser.add_argument('-channel', '--dim_channel', type=int)
parser.add_argument('-hs', '--hidden_size', type=int)
args = parser.parse_args()
if args.data_size == 0: args.data_size = None
if args.kernel_wins is not None:
    args.kernel_wins = [int(i) for i in args.kernel_wins.split(',')]

class TrainingAguments():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define training parameters
        self.training_mode = 'gpu'
        self.data_size = None
        self.batch_size = 1024
        self.epochs = 100
        self.use_early_stop = True
        self.observed_worse_val_loss_max = 10

        # define model parameters
        self.learning_rate = 0.0001
        self.hidden_size = 512
        self.emb_dim = self.hidden_size
        self.kernel_wins =[1, 2, 3, 4]
        self.dim_channel = 500
        self.num_layers=12
        

        # define log info
        self.run_name = ''
        self.run_notes = 'cnn_v0'
        self.run_tags = None,
        self.model_name = 'best_model.model'
        self.wandb = 'online'
        self.use_nni = False

        # define train data paths
        data_prefix = ''
        self.train_coarse_file = "/home/mw/input/track1_contest_4362/train/train/train_coarse.txt"
        self.train_fine_file = "/home/mw/input/track1_contest_4362/train/train/train_fine.txt"
        self.attr_to_dict_file = "/home/mw/input/track1_contest_4362/train/train/attr_to_attrvals.json"
        self.test_file = data_prefix + '/home/mw/input/track1_contest_4362/semi_testA.txt'
        # define ouput paths
        project_prefix = '/home/mw/project/'
        self.result_path = './submission/'
        self.output_dir = './output/'
        self.vocab_file = "./output/vocab.json"
        
    def update(self,args):
        for k,v in args.items():
            if v is not None:
                self.__dict__.update({k:v})

config = TrainingAguments()
config.update(args.__dict__)


# for NNI
if config.use_nni == True:
    optimized_params = nni.get_next_parameter()
    config.__dict__.update(optimized_params)

trainer = Trainer(config)

vocab = Vocab(config=config, new_vocab=False, min_freq=1)
processor = Processor(vocab=vocab)
train_data, valid_data = processor.get_train_valid_v1(config.train_fine_file, config.train_coarse_file,
                                                      config.data_size)

config.vocab_size = len(vocab)
if config.model == 'Netv9':
    model = Netv9(config)
if config.model == 'Netv8':
     model = Netv8(config)
if config.model == 'Netv5':
     model = Netv5(config)   

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)


trainer.train(model=model,
              optimizer=optimizer,
              train_data=train_data,
              valid_data=valid_data,
              train_collate_fn=processor.collate_fn,
              valid_collate_fn=processor.collate_fn,

              )

print('--------end training----------')
