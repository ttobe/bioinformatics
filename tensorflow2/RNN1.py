import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

import os
import gzip
import shutil

df = pd.read_csv('/home/newuser/ML/tensorflow2/movie_data.csv', encoding='utf-8')
print(df.tail())
#단계 1: 데이터셋 만들기
target = df.pop('sentiment') #sentiment열을 꺼낸다.
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))
print(df.values, target.values)