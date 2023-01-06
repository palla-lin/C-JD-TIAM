# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/03 
"""
import json, random
def split_file():
    file_path = 'train_fine.txt'

    data = []
    for line in open(file_path, 'r', encoding='utf-8'):
        line = json.loads(line)
        data.append(line)
    valid = random.sample(data, k=4000)
    for a in valid:
        data.remove(a)
    test = random.sample(data, k=4000)
    for a in test:
        data.remove(a)

    files = [data, valid, test]
    paths = ["train.txt", "valid.txt", "test.txt"]
    for i in range(len(paths)):
        with open(paths[i], 'w', encoding='utf-8') as f:
            for line in files[i]:
                line = json.dumps(line, ensure_ascii=False)
                f.write(line + '\n')

if __name__ == '__main__':
    split_file()