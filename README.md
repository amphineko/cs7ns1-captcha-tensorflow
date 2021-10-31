# CS7NS1 Scalable Computing

This repository contains the code for the CS7NS1 Scalable Computing project.

## Model

The model and training are living in the `crnn-ctc.ipynb` notebook.

## Prediction on Raspberry Pis

To run the prediction on Raspberry Pi, use `predict_lite.py`. Pre-trained model weights are available in the [releases](https://github.com/amphineko/cs7ns1-scalable-computing/releases) on GitHub.

Input images files should be placed in `images` folder.  Additionally, a `images.csv` file should contain a list of input filenames.

The output will be written to `output_lite.csv`.

## Acknowledgements

TODO
