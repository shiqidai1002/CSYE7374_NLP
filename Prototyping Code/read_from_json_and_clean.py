# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:44:59 2018

@author: Sean
"""

import json
from nltk.tokenize import word_tokenize
import pandas as pd

filename = 'fb_earnings_call_transcripts.json'
df = pd.read_json(filename)
df['text]

lines = df['text']
tokenized_lines = []
for line in lines:
    tokenized_line = word_tokenize(line)
    tokenized_lines.append(tokenized_line)