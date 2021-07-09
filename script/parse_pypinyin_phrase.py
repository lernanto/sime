#!/usr/bin/python3 -O

import sys
import re
import unidecode


for line in sys.stdin:
    line, _, _ = line.partition('#')
    line = line.strip()
    if line:
        word, _, pinyin = line.partition(':')
        pinyin = unidecode.unidecode(re.sub('[ǖǘǚǜ]', 'v', pinyin)).replace(' ', '')
        word = word.partition('_')[0].strip()
        if pinyin and word:
            print('{}\t{}'.format(pinyin, word))