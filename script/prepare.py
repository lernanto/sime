#!/usr/bin/python3 -O

import sys
import re
import pypinyin


for line in sys.stdin:
    for sen in re.split(r'[^\u4E00-\u9EFF]+', line):
        if sen:
            pinyin = pypinyin.slug(sen, separator='')
            print('{}\t{}'.format(pinyin, sen))
