from tensorflow.keras import *
from tensorflow.keras.layers import *

default_batch_size, default_epochs, default_step = 32, 5, 100
default_height, default_width = 64, 128


def create_model(captcha_length, n_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = Input(input_shape)

    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = Conv2D(32 * 2 ** min(i, 3),
                       kernel_initializer='he_uniform',
                       kernel_size=3,
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = [Dense(n_symbols, activation='softmax', name='char_%d' % (i+1))(x)
         for i in range(captcha_length)]

    return Model(inputs=input_tensor, outputs=x)
