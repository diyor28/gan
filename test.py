import os
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

init_op = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init_op)

from keras.models import load_model

data = pd.read_csv('parkinson.csv', index_col=0)
# x_train = data.drop('status', axis=1).values.astype('float32')
x_train = data.values.astype('float32')

model = load_model('model.h5')
noise = np.random.uniform(-1., 1., size=(5, 100))

pred = model.predict(noise)*x_train.max(axis=0)
df = pd.DataFrame(data=pred, columns=data.columns)

print(df)

