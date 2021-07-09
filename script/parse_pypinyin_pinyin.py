#!/usr/bin/python3 -O

import sys
import logging
import re
import unidecode


tone_map = {
    'ā': 'a',
    'á': 'a',
    'ǎ': 'a',
    'à': 'a',
    'ō': 'o',
    'ó': 'o',
    'ǒ': 'o',
    'ò': 'o',
    'ē': 'e',
    'é': 'e',
    'ě': 'e',
    'è': 'e',
    'ī': 'i',
    'í': 'i',
    'ǐ': 'i',
    'ì': 'i',
    'ū': 'u',
    'ú': 'u',
    'ǔ': 'u',
    'ù': 'u',
    'ǖ': 'v',
    'ǘ': 'v',
    'ǚ': 'v',
    'ǜ': 'v'
}

for line in sys.stdin:
    line = line.partition('#')[0].strip()
    if line:
        try:
            code, _, pinyins = line.partition(':')
            char = chr(int(code[2:], 16))
            for pinyin in pinyins.split(','):
                pinyin = pinyin.strip()
                if pinyin:
                    pinyin = unidecode.unidecode(re.sub('[ǖǘǚǜ]', 'v', pinyin))
                    print('{}\t{}'.format(pinyin, char))

        except Exception as e:
            logging.error('{}:{}'.format(e, line))