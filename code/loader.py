# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/03
"""

import sys
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from collections import Counter
import random, math, time
import copy
import re
import itertools
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm



class Vocab:
    """
    build or reload the vocabulary
    """

    def __init__(self, config, new_vocab=False, min_freq=0):
        self.config = config
        self.vocab_file = self.config.vocab_file
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.word2index = {"[PAD]": 0, "[MASK]": 1, "[SEP]": 2, "[UNK]": 3, "[CLS]": 4}
        self.n_words = len(self.word2index)
        self.index2word = dict([(v, k) for k, v in self.word2index.items()])
        if new_vocab:
            self.build_from_scratch(vocab_path=self.vocab_file, min_freq=min_freq)
        else:
            self.build_from_json(self.vocab_file)

    def __len__(self):
        return self.n_words

    def build_from_json(self, vocab_path):
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            print(f'read vocabulary from:{vocab_path}')
            self.word2index = vocab
            self.index2word = dict([(v, k) for k, v in self.word2index.items()])
            self.n_words = len(self.word2index)
        else:
            print(f'error, no such vocabulary path:{vocab_path}')
            exit()
        print(f'vocab_size:{self.n_words}')

    def build_from_scratch(self, vocab_path, min_freq):
        # self._add_word('a0')
        # self.attr_dict, self.ori_attr_dict = load_attr_dict(attr_dict_path)
        # all_vars = itertools.chain.from_iterable(self.attr_dict.values())
        # self.attrvals = "|".join(all_vars)
        # for i, (k, vals) in enumerate(self.attr_dict.items()):
        #     for v in vals:
        #         self.attr_type[v] = 'a' + str(i + 1)
        #         self._add_word('a' + str(i + 1))
        counter = Counter()
        raw_coarse_data = [self.config.train_coarse_file]
        raw_fine_data = [self.config.train_fine_file]
        for path in raw_coarse_data:
            print(path)
            for line in tqdm(open(path, 'r', encoding='utf-8')):
                try:
                    line = json.loads(line)
                except:
                    continue
                counter.update(line['title'])
        for path in raw_fine_data:
            print(path)
            for line in tqdm(open(path, 'r', encoding='utf-8')):
                line = json.loads(line)
                counter.update(line['title'])
                for k, v in line['key_attr'].items():
                    counter.update(k)
                    counter.update(v)

        for k, v in counter.items():
            if v > min_freq:
                self._add_word(k)
        for i in range(13):
            self._add_word('a' + str(i))
        print('vocab size:', self.n_words)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.word2index, f, ensure_ascii=False)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def seq_to_index(self, sequence):
        # for word in sequence:
        #     if word not in self.word2index:
        #         print(word)
        indices = [self.word2index[word] if word in self.word2index else self.word2index['[UNK]'] for word in sequence]
        return indices

    def index_to_seq(self, indices):
        sequence = [self.index2word[index] for index in indices]
        return sequence


class Processor:
    "form a batch, tokenize and convert to id, "

    def __init__(self, vocab):
        self.vocab = vocab
        self.attr_dict, self.ori_attr_dict = self._load_attr_dict(self.vocab.config.attr_to_dict_file)
        self.fake_attr_map = self._load_fake_attr_map()
        self.same_attr_map = self._load_same_attr_map()
        self.attr_type_map, self.key_type_map = self._load_attr_type_map()
        self.attr_key_map = self._load_attr_key_map()
        self.all_vars = list(itertools.chain.from_iterable(self.attr_dict.values()))
        self.titles = []

    def _load_attr_dict(self, filepath):
        # 读取属性字典
        with open(filepath, 'r') as f:
            attr_dict = {}
            ori_attr_dict = {}
            for attr, attrval_list in json.load(f).items():
                ori_attr_dict[attr] = attrval_list
                attrval_list = list(map(lambda x: x.split('='), attrval_list))
                attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
        return attr_dict, ori_attr_dict

    def _load_fake_attr_map(self):
        fake_attr_map = defaultdict(list)
        attr_dict, ori_attr_dict = self.attr_dict, self.ori_attr_dict
        for k, vars in attr_dict.items():
            cans = ori_attr_dict[k]
            for var in vars:
                for can in cans:
                    if var in can:
                        pass
                    else:
                        c = can.split('=')
                        fake_attr_map[var].extend(c)

        return fake_attr_map

    def _load_same_attr_map(self):
        same_attr_map = {}
        attr_dict, ori_attr_dict = self.attr_dict, self.ori_attr_dict
        for k, vars in attr_dict.items():
            cans = ori_attr_dict[k]
            for var in vars:
                for can in cans:
                    if var in can:
                        c = can.split('=')
                        same_attr_map[var] = c[0]
        return same_attr_map

    def _load_attr_type_map(self):
        attr_type_map = {}
        attr_dict, ori_attr_dict = self.attr_dict, self.ori_attr_dict
        keys = list(attr_dict.keys())

        temp_map = defaultdict()
        for i in range(13):
            temp_map[keys[i - 1]] = 'a' + str(i)
        temp_map.update({'图文': 'a13'})
        self.vocab._add_word('a13')
        for k, values in attr_dict.items():
            for v in values:
                attr_type_map[v] = temp_map[k]
        return attr_type_map, temp_map

    def _load_attr_key_map(self):
        attr_key_map = {}
        for key, attrs in self.attr_dict.items():
            for attr in attrs:
                if attr not in attr_key_map:
                    attr_key_map.update({attr: key})
                else:
                    attr_key_map.update({'鞋' + attr: key})
        return attr_key_map

    def _get_example(self, img, query_type, query_text, label, feature):
        labels = label
        # if label==0:
        #     labels=[1.0,0.0]
        # if label == 1:
        #     labels=[0.0,1.0]

        example = {
            'img': img,
            'query_type': query_type,
            'query_text': query_text,
            'feature': feature,
            'labels': labels
        }
        return example

    def match_attrval(self, title, attr, attr_dict):
        # 在title中匹配属性值
        attrvals = "|".join(attr_dict[attr])
        ret = re.findall(attrvals, title)
        # return "{}{}".format(attr, ''.join(ret))
        return ''.join(list(set(ret)))

    @classmethod
    def _pad(cls, batch):
        lengths = [len(seq) for seq in batch]
        max_length = max(lengths)
        for seq in batch:
            while len(seq) < max_length:
                seq.append(0)
        return batch

    def get_dataset(self, data, data_size=None):
        print('tokenizing.....')
        random.shuffle(data)
        for item in data:
            # query_type = self.tokenize(item['query_type'])
            # for k,v in query_type.items():
            # k='q_'+k
            # item.update({k:v})
            text = item['query_text']  # item['query_type']+
            text = self.remove_stopwords(text)
            tokenized_text = self.tokenize(text, query_type=item['query_type'])
            item.update(tokenized_text)
        return data

    def collate_fn(self, data):

        batch = defaultdict()
        data_keys = list(data[0].keys())
        data_values = []
        for example in data:
            data_values.append(example.values())
        for i, v in enumerate(zip(*data_values)):
            try:
                v = self._pad(v)
            except:
                pass
            try:
                v = torch.tensor(v)
            except:
                if data_keys[i] == 'labels':
                    v = None
            batch[data_keys[i]] = v
        batch = dict(batch)

        return batch

    def tokenize(self, text, query_type=None):
        tokens = [self.vocab.cls_token] + [w for w in text] + [self.vocab.sep_token]
        ret = re.finditer("|".join(self.all_vars), text)
        query_type_ids = [self.key_type_map[query_type] for i in tokens]
        attr_type = ['a0' for i in tokens]
        for match in ret:
            for i in range(match.span()[0] + 1, match.span()[1] + 1):
                attr_type[i] = self.attr_type_map[match.group()]

        tokens = self.vocab.seq_to_index(tokens)
        attr_type = self.vocab.seq_to_index(attr_type)
        query_type_ids = self.vocab.seq_to_index(query_type_ids)

        return {'input_ids': tokens, 'type_ids': attr_type, 'query_type_ids': query_type_ids}

    def merge_attr(self, text):
        def func(match):
            return self.same_attr_map[match.group(0)]

        text = re.sub(pattern="|".join(self.all_vars), repl=func, string=text)
        return text

    def remove_stopwords(self, string):
        newstring = re.sub(r'[0-9年;/a-z]+', '', string)
        return newstring

    def preprocess_fine_data(self, filepath=None, data=None, data_size=None):
        positive_title = []
        positive_attr = []
        if data is None:
            f = open(filepath, 'r', encoding='utf-8')
        else:
            f = data
        for n, line in enumerate(f):
            if data_size is not None and n == data_size: break
            if isinstance(line, str):
                line = json.loads(line)
            assert line['match']['图文'] == 1
            self.titles.append(line['title'])
            example = self._get_example(img=line['img_name'],
                                        query_type='图文',
                                        query_text=line['title'],
                                        feature=line['feature'],
                                        label=1)
            positive_title.append(example)
            for k, v in line['key_attr'].items():
                example = self._get_example(img=line['img_name'],
                                            query_type=k,
                                            query_text=v,
                                            feature=line['feature'],
                                            label=1)
                positive_attr.append(example)
        return positive_title, positive_attr

    def preprocess_coarse_data(self, filepath=None, data=None, data_size=None):
        positive = []
        negative = []
        if data is None:
            f = open(filepath, 'r', encoding='utf-8')
        else:
            f = data
        for n, line in enumerate(f):
            if data_size is not None and n == data_size: break
            if isinstance(line,str):
                line = json.loads(line)
            self.titles.append(line['title'])
            if line['match']['图文'] == 1:

                example = self._get_example(img=line['img_name'],
                                            query_type='图文',
                                            query_text=line['title'],
                                            label=1,
                                            feature=line['feature'], )
                positive.append(example)
            elif line['match']['图文'] == 0:
                example = self._get_example(img=line['img_name'],
                                            query_type='图文',
                                            query_text=line['title'],
                                            feature=line['feature'],
                                            label=0)
                negative.append(example)

        return positive,  negative

    def pp_v1(self, positive, negative, weight=None):

        for item in positive:
            fake_title = self.replace_attr(item['query_text'], item['query_type'], weight=weight)
            if fake_title is not None:
                fake_item = self._get_example(img=item['img'], query_text=fake_title, query_type=item['query_type'],
                                              feature=item['feature'], label=0)
                negative.append(fake_item)
            # if len(negative)==len(positive):
            #     break

        data = positive + negative
        P = len(positive)
        N = len(negative)
        R = round((P / N), 3)
        print(f'Total:{P + N}, Pos:{P}, Neg:{N}, P/N:{R} P/(P+N):{P / (P + N)}')
        return data

    def get_train_valid_v1(self, fine_file=None, coarse_file=None, data_size=None):
        if fine_file is None:
            fine_file = '/home/mw/input/track1_contest_4362/train/train/train_fine.txt'
        if coarse_file is None:
            coarse_file = '/home/mw/input/track1_contest_4362/train/train/train_coarse.txt'
        d=False
        positive_title_extra=[]
        if d:
            det_list = []
            det_count = 0
            for line in open('./output/det.json', 'r', encoding='utf-8'):
                det_list.append(line.rstrip('\n'))
            fine_data = []
            
            for line in open(fine_file, 'r', encoding='utf-8'):
                line = json.loads(line)
                if line['img_name'] not in det_list:
                    fine_data.append(line)
                else:
                    det_count += 1
                    if line['match']['图文'] == 1:
                        example = self._get_example(img=line['img_name'],
                                                    query_type='图文',
                                                    query_text=line['title'],
                                                    label=1,
                                                    feature=line['feature'], )
                        positive_title_extra.append(example)
            print('deleted items', det_count)
            train_data, valid_data = train_test_split(fine_data, test_size=0.12, random_state=3334)
        else:
            data = []
            for line in open(fine_file, 'r', encoding='utf-8'):
                data.append(line)
            train_data, valid_data = train_test_split(data, test_size=0.1, random_state=3334)

        # train
        positive_title_fine, positive_attr_fine = self.preprocess_fine_data(data=train_data, data_size=data_size)
        positive_title_coarse,  negative_title_coarse = self.preprocess_coarse_data(coarse_file,data_size=data_size)
        positive_title = positive_title_coarse + positive_title_fine + positive_title_extra
        positive_attr = positive_attr_fine
        positive = positive_title + positive_attr

        negative = negative_title_coarse

        train_data = self.pp_v1(positive, negative)
        # test
        positive_title_fine, positive_attr_fine = self.preprocess_fine_data(data=valid_data, data_size=data_size)
        positive = positive_title_fine + positive_attr_fine
        negative = []
        valid_data = self.pp_v1(positive, negative)
        train_data = self.get_dataset(train_data)
        valid_data = self.get_dataset(valid_data)
        return train_data, valid_data


    def process_test_file(self, file_path):
        data = []
        for n, line in enumerate(open(file_path, 'r', encoding='utf-8')):
            line = json.loads(line)
            for query_type in line['query']:
                text = line['title'] if query_type == '图文' else self.match_attrval(line['title'], query_type,
                                                                                   self.attr_dict)

                example = self._get_example(img=line['img_name'], query_text=text, query_type=query_type,
                                            feature=line['feature'], label=None)
                data.append(example)
        data = self.get_dataset(data)
        return data

    def replace_attr(self, text, qtype, weight=None):
        def func(match):
            fake = self.fake_attr_map[match.group(0)]
            return random.sample(population=fake, k=1)[0]

        if weight is None:
            weight = [0.05, 0.85, 0.1] 
        if qtype == '图文':
            mode = random.choices([0, 1, 2], weights=weight, k=1)[0]
            vals = re.findall(pattern="|".join(self.all_vars), string=text)
            if len(vals) == 0:
                return None
            if mode == 0:
                # 多个位置
                fake_text = re.sub(pattern="|".join(vals), repl=func, string=text)
            elif mode == 1:
                # 单个位置
                p = random.sample(population=vals, k=1)[0]
                r = random.sample(population=self.fake_attr_map[p], k=1)[0]
                fake_text = re.sub(pattern=p, repl=r, string=text)
            else:
                fake_text = random.sample(population=self.titles, k=1)[0]
            return fake_text
        else:
            fake = self.fake_attr_map[text]
            return random.sample(population=fake, k=1)[0]




if __name__ == '__main__':
    pass
