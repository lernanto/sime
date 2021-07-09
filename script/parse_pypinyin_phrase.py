#!/usr/bin/python3 -O

import sys
import unidecode


for line in sys.stdin:
    if not line.startswith('#'):
        line = line.strip()
        if line:
            word, _, pinyin = line.partition(':')
            print('{}\t{}'.format(unidecode.unidecode(pinyin).replace(' ', ''), word))
