import re
import numpy as np
import os
from text.phones_mix import phone_to_id

def label_to_sequence(label):
    phones = []

    with open(label, 'r', encoding='utf8') as lab_reader:
        line = lab_reader.readline()
        meta = line.strip().split('|')[2].split(' ')
        for item in meta:
            if item not in phone_to_id.keys():
                continue
            phones.append(item)

    phones.append("~")
    print(phones)
    return [phone_to_id[p] for p in phones]

def label_to_sequence2(pyin):
    phones = []
    pyin = pyin.split(' ')
    print(pyin)
    for item in pyin:
        #print('item: ', item)
        if item not in phone_to_id.keys():
            #print(item)
            continue
        phones.append(item)

    phones.append("~")
    #print(phones)
    return [phone_to_id[p] for p in phones]
