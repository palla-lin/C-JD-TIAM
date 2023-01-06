# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/03 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from loguru import logger

import json
import random
from loader import Vocab, Processor, DataLoader
from model import Netv4, Netv5,Netv6, Netv8, Netv7,Netv9
from tqdm import tqdm
from config import config

torch.manual_seed(1234)
random.seed(1234)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate(model, loader, vocab, num=10000):
    def tensor_to_seq(tensor):
        if tensor.size(0) == 1:
            pass
        else:
            tensor = tensor.transpose(1, 0)
        index = tensor.tolist()[0]
        seq = vocab.index_to_seq(index)
        return seq

    model.eval()
    all_pred = {}
    with torch.no_grad():
        for n, input in enumerate(loader):
            if n >= num:
                break
       
            img = input['img']
            query_types = input["query_type"]
            query_text = input["query_text"]
            
            output = model(**input)
            output = output['outputs']
            out = F.softmax(output, dim=-1).view(-1, 2)
            preds = torch.argmax(out, dim=-1, keepdim=True).tolist()

            for i in range(output.size(0)):
                query_type = query_types[i]
                pred = preds[i][0]
                img_name = img[i]
                predicted_results[img_name]['match'][query_type] = pred
                # if img[i] not in all_pred:
                #     all_pred[img[i]] = {'match':
                #                             {query_type: pred}
                #                         }
                # else:
                #     all_pred[img[i]]['match'].update({query_type: pred})
    return all_pred


def get_answer_template(filepath):
    from collections import defaultdict
    def fun():
        return defaultdict(dict)

    template = defaultdict(fun)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            for q in line['query']:
                template[line['img_name']]['match'][q] = None
    return template


def save(data, path):
    rets = []
    for k, v in data.items():
        ret = {
            "img_name": k,
            "match": v['match']
        }
        rets.append(json.dumps(ret, ensure_ascii=False) + '\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(rets)
        print(f'saved {path}')

class Config():
    def __init__(self):
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='proj')

    parser.add_argument('-M', '--model_path', type=str)
    parser.add_argument('-T', '--test_file', type=str, default= '/home/mw/input/track1_contest_4362/semi_testA.txt')
    parser.add_argument('-R', '--result_file', type=str,default='/submission/result_v1.txt')
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-num', '--num', type=int, default=0)
    args = parser.parse_args()

    result_path = args.result_file
    model_path =args.model_path
    test_file =args.test_file

    vocab = Vocab(config=config, new_vocab=False)
    processor = Processor(vocab)

    predicted_results = get_answer_template(test_file)
    
    
    data = processor.process_test_file(test_file)
    loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, collate_fn=processor.collate_fn)

    model_CKPT = torch.load(model_path )
    config = model_CKPT['config']
    if isinstance(config, dict):
        con = Config()
        con.__dict__.update(config)
        model = Netv5(con).to(device)
    else:
        config.num_layers=12
        model = Netv5(config).to(device)
    model.load_state_dict(model_CKPT['state_dict'])
   
    
    if args.num ==0:
        num=10000000
    else:
        num = args.num
    generate(model, loader, vocab, num=num)
    save(predicted_results, path=result_path)
