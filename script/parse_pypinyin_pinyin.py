#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

'''
把 python-pinyin 原始文本格式的字拼音数据转成输入法引擎需要的格式.

原始字典数据见 https://github.com/mozillazg/pinyin-data
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
            code, _, pinyins = line.partition(':')
            char = chr(int(code[2:], 16))
            for pinyin in pinyins.split(','):
                pinyin = pinyin.strip()
                if pinyin:
                    pinyin = unidecode.unidecode(re.sub('[üǖǘǚǜ]', 'v', pinyin))
                    print('{}\t{}'.format(pinyin, char))

    except Exception as e:
        logging.error('{}:{}'.format(e, line))