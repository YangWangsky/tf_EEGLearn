
# tf_EEGLearn
All implementations are not guaranteed to be correct, have not been checked by original authors, only reimplemented from the paper description and open source code from original authors.


## Prerequisites

- python >= 3.5
- tensorflow-gpu >= 1.2
  - `conda install tensorflow-gpu=1.2`
- Numpy
- Scipy
- Scikit-learn

## EEGLearn

It's a tensorflow implementation for EEGLearn

    Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).
    http://arxiv.org/abs/1511.06448

For more information please see https://github.com/pbashivan/EEGLearn

<center>

![Diagram](./images/diagram.png 'Diagram.png')

![AEP](./images/AEP.png 'AEP.png') 
</center>

## Leave-one-subject-out Experments

### Model: EEGLearn


| Subject id | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | mean |
|---         |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---  |---  |---  |---  |
| CNN test acc | 55.14 | 73.58 | 93.47 | 100.00 | 100.00 | 99.50 | 99.48 | 100.00 | 98.57 | 100.00 | 98.62 | 72.25 | 49.55 | 87.70 |