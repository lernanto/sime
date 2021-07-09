#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

'''
把 python-pinyin 原始文本格式的词拼音数据转成输入法引擎需要的格式.

原始词典数据见 https://github.com/mozillazg/phrase-pinyin-data
'''

__author__ = '黄艺华'


import sys
import logging
import re
import unidecode


for line in sys.stdin:
    try:
        line = line.partition('#')[0].strip()
        if line:
            word, _, pinyin = line.partition(':')
            pinyin = unidecode.unidecode(re.sub('[üǖǘǚǜ]', 'v', pinyin)).replace(' ', '')
            word = word.partition('_')[0].strip()
            if pinyin and word:
                print('{}\t{}'.format(pinyin, word))

    except Exception as e:
        logging.error('{}:{}'.format(e, line))