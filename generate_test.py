#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from config import symbols
from train import CaptchaSequence


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([symbols[x] for x in y])


data = CaptchaSequence(symbols, batch_size=1, steps=1)
X, y = data[0]
plt.imshow(X[0])
plt.title(decode(y))
plt.show()
