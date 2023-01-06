# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/05 
"""
import os
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sm
import random, time

import wandb
from tqdm import tqdm


class TrainingAguments():
    def __init__(self):
        self.training_mode = 'gpu'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define training parameters
        self.hidden_size = 512

        # define model parameters
        self.data_size = None
        self.batch_size = 1024
        self.learning_rate = 0.0001
        self.epochs = 1000

        self.use_early_stop = True
        self.observed_worse_val_loss_max = 10


        # define log info
        self.run_name = ''
        self.run_notes = 'v1,net9,cnn,'
        self.run_tags = None,
        self.wandb = 'disabled'
        self.use_nni = True

        # define paths
        data_prefix = ''
        self.train_coarse_file = "./data/train_coarse.txt"
        self.train_fine_file = "./data/train_fine.txt"
        self.attr_to_dict_file = "./data/attr_to_attrvals.json"
        self.test_file = data_prefix + 'semi_testA.txt'
        project_prefix = '/home/mw/project/'
        self.result_path = project_prefix + 'submission/'
        self.output_dir = project_prefix + 'output/'
        self.vocab_file = "./data/vocab.json"
        self.model_name = 'best_model.model'

class Trainer():
    def __init__(self, config, ):
        self.config = config
        self.training_mode = config.training_mode
        self.observed_worse_val_loss_max = config.observed_worse_val_loss_max

        self.output_dir, self.path = self._get_path(config.output_dir, config.model_name)
        self._init_seed()
        self._init_env()

    def _get_path(self, output_dir, model_name):
        H = int(time.strftime(f"%H", time.localtime())) + 2
        output_dir = output_dir + '/runs/' + time.strftime(f"%b%d_{H}-%M-%S", time.localtime())
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        path = output_dir + '/' + model_name

        return output_dir, path

    def _init_seed(self):
        seed = 1234
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def _init_env(self):
        self.local_rank=0
        if self.training_mode == 'dist':
            # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

            os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            # torch.distributed.init_process_group(backend="nccl",init_method='tcp://127.0.0.1:23456',world_size=1,rank=local_rank)

        if self.local_rank == 0:
            os.environ['WANDB_MODE'] = self.config.wandb
            wandb.init(
                project="jd",
                notes=self.config.run_notes,
                # tags = config.run_tags,
                save_code=True,
            )
            # save all codes
            for p in ['code/' + i for i in os.listdir('code/')]:
                wandb.save(p)
            print('save all codes.....')
        if self.config.use_nni:
            import nni


    def _count_parameters(self, model):
        num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model has {num_para} trainable parameters')

    def _save_model(self, config, model, optimizer, path):
        try:
            state_dict = model.module.state_dict()
        except:
            state_dict = model.state_dict()
        torch.save({'config': config.__dict__, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}, path)
        wandb.save(path)
        print(f'Saved model...', path)

    def train(self, model, optimizer, train_data, valid_data, valid_collate_fn, train_collate_fn):
        self._count_parameters(model)
        self.train_sampler = None
        if self.training_mode == 'dist':
            valid_loader = DataLoader(valid_data, batch_size=self.config.batch_size, shuffle=False,
                                      collate_fn=valid_collate_fn)
            self.train_sampler = DistributedSampler(train_data)
            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, sampler=self.train_sampler,
                                      collate_fn=train_collate_fn)

            model.to(self.config.device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank],
                                                              output_device=self.local_rank)
        else:
            valid_loader = DataLoader(valid_data, batch_size=self.config.batch_size, shuffle=False,
                                      collate_fn=valid_collate_fn)
            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True,
                                      collate_fn=train_collate_fn)
            model.to(self.config.device)


        self._train(model, optimizer, train_loader, valid_loader)
        # r.conclude()

    def _train(self, model, optimizer, train_loader, valid_loader):
        best_score = float('-inf')
        observed_worse_val_loss = 0

        for epoch in tqdm(range(1, self.config.epochs + 1)):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            # weights = [[0.1, 0.7, 0.2],]
            # weight = random.sample(weights,k=1)[0]
            # print('Weights:',weight)
            # train_data = processor.re_gen_train(train_loader_pp, weight=weight)

            train_loss = self.train_epoch(model, optimizer, train_loader)


            if self.local_rank == 0:
                val_loss, val_acc = self.eval_epoch(model, valid_loader)

                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})
                print(f'\nepoch:{epoch}, train_loss:{train_loss}, val_loss:{val_loss},val_acc:{val_acc}')



                score = val_acc
                if self.config.use_nni:
                    nni.report_intermediate_result(score)
                if self.config.use_early_stop:
                    if score > best_score:
                        best_score = score
                        observed_worse_val_loss = 0
                        self._save_model(config=self.config, model=model, optimizer=optimizer, path=self.path)
                        with open(self.output_dir + '/checkpoint.txt', 'w', encoding='utf-8') as f:
                            f.write(f'epoch:{epoch}, best_score:{best_score}, val_loss:{val_loss}')
                    else:
                        observed_worse_val_loss += 1

                else:
                    self._save_model(config=self.config, model=model, optimizer=optimizer, path=self.path)
            # time.sleep(0.003)
            if self.training_mode == 'dist':
                torch.distributed.barrier()

            if observed_worse_val_loss >= self.observed_worse_val_loss_max:
                if self.config.use_nni:
                    nni.report_final_result(score)
                print(f'Have observed successively {self.observed_worse_val_loss_max}'
                      f' worse validation results.\nStop training...')
                break

    def train_epoch(self, model, optimizer, train_loader):

        model.train()
        train_loss = 0.
        for i, input in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**input)
            loss = outputs["loss"]
            # loss = loss.sum().mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / (i + 1)

    def eval_epoch(self, model, loader):

        eval_loss = 0.
        eval_acc = 0.
        model.eval()
        with torch.no_grad():
            all_pred = []
            all_label = []
            for i, input in enumerate(loader):
                outputs = model(**input)
                loss = outputs["loss"]
                # loss = loss.sum().mean()
                eval_loss += loss.item()
                output = outputs['outputs']

                out = torch.softmax(output, dim=-1)
                pred = torch.argmax(out, dim=-1).reshape(-1)
                all_pred.append(pred)
                all_label.append(input['labels'].reshape(-1))

            acc = self.compute_acc(all_pred, all_label)

        return eval_loss / (i + 1), acc

    def compute_acc(self, pred, gold):

        pred = torch.cat(pred, dim=-1).to('cpu')
        gold = torch.cat(gold, dim=-1).to('cpu')

        matrix = sm.confusion_matrix(y_true=gold, y_pred=pred)
        print(matrix)
        # report = sm.classification_report(y_true=gold, y_pred=pred, target_names=['N', 'Y'], )
        # print(report)
        acc = torch.sum((pred == gold)) / gold.size(0)
        return acc.item()