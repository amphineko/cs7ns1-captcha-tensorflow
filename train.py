#!/usr/bin/env python3

import argparse
import multiprocessing
import random
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from captcha.image import ImageCaptcha
from tensorflow.keras.utils import Sequence

from model import (create_model, default_batch_size, default_epochs,
                   default_height, default_width)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CaptchaSequence(Sequence):
    def __init__(self,
                 captcha_length,
                 captcha_symbols,
                 batch_size,
                 generator,
                 width=128,
                 height=64):
        self.batch_size = batch_size
        self.generator = generator
        self.height = height
        self.n_class = len(captcha_symbols)
        self.n_len = captcha_length
        self.symbols = captcha_symbols
        self.width = width

    def __len__(self):
        return self.batch_size

    def __getitem__(self, _):
        X = np.zeros((self.batch_size, self.height, self.width, 3),
                     dtype=np.float32)
        y = [
            np.zeros((self.batch_size, self.n_class), dtype=np.uint8)
            for _ in range(self.n_len)
        ]

        for i in range(self.batch_size):
            text = ''.join(
                [random.choice(self.symbols) for _ in range(self.n_len)])
            image = self.generator.generate_image(text)

            X[i] = np.array(image) / 255.0
            for j, ch in enumerate(text):
                y[j][i, :] = 0
                y[j][i, self.symbols.find(ch)] = 1

        return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--height', default=default_height, type=int)
    parser.add_argument('--width', default=default_width, type=int)

    parser.add_argument('--batch-size', default=default_batch_size, type=int)
    parser.add_argument('--epochs', default=default_epochs, type=int)
    parser.add_argument('--validation-size',
                        default=default_batch_size,
                        type=int)

    parser.add_argument('--captcha-length', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--symbols', default='symbols.txt', type=str)

    parser.add_argument('--gpu', type=bool)

    args = parser.parse_args()

    batch_size, epochs, validation_size = args.batch_size, args.epochs, args.validation_size

    symbols_path = Path(args.symbols)
    if not symbols_path.is_file():
        raise IOError('Symbol file is not exist: ' + symbols_path)
    symbols = symbols_path.read_text()

    checkpoint_path = Path(args.model_name).with_suffix('.h5')
    logs_path = Path(args.model_name).with_suffix('.csv')
    resume_path = Path(args.model_name).with_suffix('.resume.h5')

    captcha_generator = ImageCaptcha(width=128, height=64)

    if args.gpu:
        device = tf.device('/device:GPU:0')

        # tensorflow gpu memory fix
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        device = tf.device('/device:CPU:0')

    with device:
        model = create_model(args.captcha_length, len(symbols),
                             (args.height, args.width, 3))

        if checkpoint_path.is_file():
            print('Loading checkpint from ' + checkpoint_path.name)
            model.load_weights(checkpoint_path)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        training_data = CaptchaSequence(args.captcha_length,
                                        symbols,
                                        batch_size=batch_size,
                                        generator=captcha_generator,
                                        height=args.height,
                                        width=args.width)
        validation_data = CaptchaSequence(args.captcha_length,
                                          symbols,
                                          batch_size=validation_size,
                                          generator=captcha_generator,
                                          height=args.height,
                                          width=args.width)

        callbacks = [
            keras.callbacks.EarlyStopping(patience=3),
            keras.callbacks.CSVLogger(logs_path, append=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_best_only=True)
        ]

        model.fit(training_data,
                  callbacks=callbacks,
                  epochs=epochs,
                  use_multiprocessing=True,
                  validation_data=validation_data,
                  workers=multiprocessing.cpu_count())
