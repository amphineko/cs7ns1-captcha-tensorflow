#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pandas as pd
import string
from tflite_runtime.interpreter import Interpreter
import time


symbols = ' ' + ''.join(sorted(string.ascii_uppercase + string.ascii_lowercase + string.digits + '#%-:<>[]{}'))
decode_label = lambda s: ''.join([symbols[x] for x in s[:s.index(0)]])


def decode_ctc_lite(logits, symbols):
    output, last_logit = [], None
    for logit in logits.argmax(axis=1):
        if (logit < len(symbols)) and (logit != last_logit):
            output.append(logit)
        last_logit = logit

    return output


def batch_predict_lite():
    # lite model setup

    interpreter = Interpreter(model_path=str('./ctc-lite.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # read inference list

    parent_path = os.path.abspath('./images')

    df = pd.read_csv('./images.csv', header=None, index_col=False, names=['filename'])[['filename']]
    df['result'] = ''

    start_time = time.time()
    for idx, row in df.iterrows():
        filename = row['filename']
        filename = os.path.join(parent_path, filename)

        x = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # = float64
        x = np.array(x, dtype=np.float32) / 255.0       # = float32
        x = np.expand_dims(x, axis=-1)                  # = (128, 64)
        x = x.transpose(1, 0, 2)                        # = (64, 128, 1)
        x = np.expand_dims(x, axis=0)                   # = (1, 128, 64, 1)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])

        y_pred = decode_label(decode_ctc_lite(y_pred[0], symbols))

        # print('%s: %s' % (os.path.basename(filename), y_pred))

        row['result'] = y_pred

        if idx % 100 == 0:
            print('predicted %d of %d images in %f seconds' % (idx, df.shape[0], time.time() - start_time))

    print('predicted %d images in %f seconds' % (df.shape[0], time.time() - start_time))
    
    df.sort_values(by=['filename'], ascending=True)
    df.to_csv('./output_lite.csv', columns=['filename', 'result'], header=False, index=False)


batch_predict_lite()
