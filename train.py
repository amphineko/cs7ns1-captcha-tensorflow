#!/usr/bin/env python3

import multiprocessing
import random

import numpy as np
from captcha.image import ImageCaptcha
from tensorflow.keras.utils import Sequence

from config import max_len, min_len, symbols
from model import model

len_range = range(min_len, max_len + 1)
generator = ImageCaptcha(width=128, height=64)


class CaptchaSequence(Sequence):
    def __init__(self, symbols, batch_size, steps, width=128, height=64):
        self.batch_size = batch_size
        self.generator = generator
        self.height = height
        self.n_class = len(symbols)
        self.steps = steps
        self.symbols = symbols
        self.width = width

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        n_len = random.choice(len_range)

        X = np.zeros((self.batch_size, self.height, self.width, 3),
                     dtype=np.float32)
        y = [
            np.zeros((self.batch_size, self.n_class), dtype=np.uint8)
            for _ in range(max_len)
        ]

        for i in range(self.batch_size):
            text = ''.join([
                random.choice(self.symbols)
                for _ in range(n_len)
            ])
            image = self.generator.generate_image(text.ljust(max_len))

            X[i] = np.array(image) / 255.0
            for j, ch in enumerate(text):
                y[j][i, :] = 0
                y[j][i, self.symbols.find(ch)] = 1

        return X, y


if __name__ == '__main__':
    from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                            ModelCheckpoint)
    from tensorflow.keras.optimizers import *

    train_data = CaptchaSequence(symbols, batch_size=128, steps=1000)
    valid_data = CaptchaSequence(symbols, batch_size=128, steps=100)
    callbacks = [
        EarlyStopping(patience=3),
        CSVLogger('model.csv'),
        ModelCheckpoint('model_best.h5', save_best_only=True)
    ]

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=Adam(1e-3, amsgrad=True))
    model.fit_generator(train_data,
                        callbacks=callbacks,
                        epochs=100,
                        use_multiprocessing=False,
                        validation_data=valid_data,
                        workers=multiprocessing.cpu_count())
