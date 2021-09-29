from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from config import max_len, symbols

width, height, n_len, n_class = 128, 64, max_len, len(symbols)

inputs = Input((height, width, 3))
x = inputs
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32 * 2 ** min(i, 3),
                   kernel_initializer='he_uniform',
                   kernel_size=3,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class,
           activation='softmax',
           name='c%d' % (i+1)
           )(x) for i in range(n_len)]

model = Model(inputs=inputs, outputs=x)
