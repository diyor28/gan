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

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


class GAN:
    def __init__(self, path_to_data='parkinson.csv'):
        self.output_size = 23
        # self.output_size = 22
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.adversarial = self.build_adversarial()
        data = pd.read_csv(path_to_data, index_col=0)
        # self.x_train = data.drop('status', axis=1).values.astype('float32')
        self.x_train = data.values.astype('float32')
        self.x_train = self.x_train/self.x_train.max(axis=0)

    def build_generator(self):
        input_layer = Input(shape=(100, ))
        dropout = 0.4

        hidden = Dense(64, activation='relu')(input_layer)
        hidden = Dropout(dropout)(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        hidden = Dropout(dropout)(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        hidden = Dropout(dropout)(hidden)
        output = Dense(self.output_size, activation='relu')(hidden)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
        return model

    def build_discriminator(self):
        dropout = 0.4
        input_layer = Input(shape=(self.output_size, ))

        hidden = Dense(64, activation='relu')(input_layer)
        hidden = Dropout(dropout)(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        hidden = Dropout(dropout)(hidden)
        output = Dense(2, activation='softmax')(hidden)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_adversarial(self):
        output = self.discriminator(self.generator.output)
        model = Model(inputs=[self.generator.input], outputs=[output])
        model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def save_model(self):
        self.generator.save('model.h5')

    def train(self, train_steps=10_000, batch_size=16, verbose=1):
        for i in range(train_steps):
            train_data = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size)]
            noise = np.random.uniform(-1., 1., size=(batch_size, 100))
            fake = self.generator.predict(noise)
            x = np.concatenate((train_data, fake))
            y = np.zeros((2*batch_size, 2))

            y[:batch_size, 0] = 1
            y[batch_size:, 1] = 1

            disc_loss = self.discriminator.train_on_batch(x, y)

            y = np.zeros((batch_size, 2))
            y[:, 0] = 1

            noise = np.random.uniform(-1., 1., size=(batch_size, 100))

            adv_loss = self.adversarial.train_on_batch(noise, y)

            if verbose == 1:
                print("Adversarial loss: {} | Discriminator loss: {}".format(adv_loss, disc_loss))


if __name__ == '__main__':
    network = GAN()
    network.train()
    network.save_model()