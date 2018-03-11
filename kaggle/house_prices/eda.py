import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# set word dir
import os
from os import path
import sys
os.listdir(".")
cur_dir = path.dirname(__file__)
sys.path.append(cur_dir)

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
print(train.head())
print(test.head())

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']