#!/usr/bin/python3 -O

import sys
import logging
import unidecode


for line in sys.stdin:
    if not line.startswith('#'):
        line = line.strip()
        if line:
            try:
                char, _, pinyin = line.partition(':')
                char = chr(int(char[2:], 16))
                pinyin = unidecode.unidecode(pinyin.partition('#')[0]).replace(' ', '')
                print('{}\t{}'.format(pinyin, char))
            except Exception as e:
                logging.error('{}:{}'.format(e, line))
