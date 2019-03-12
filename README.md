
# Notes
All implementations are not guaranteed to be correct, have not been checked by original authors, only reimplemented from the paper description.


# EEGLearn

It's a tensorflow implementation for EEGLearn
    
    Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).
    http://arxiv.org/abs/1511.06448

For more information please see https://github.com/pbashivan/EEGLearn

<center>

![Diagram](./images/diagram.png 'Diagram.png')

![AEP](./images/AEP.png 'AEP.png') 
<center>

# Leave-one-subject-out Experments

## Model: EEGLearn


| Subject id | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | mean |
|---         |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---  |---  |---  |---  |
| CNN test acc | 0.5243 | 0.7028 | 0.9397 | 0.995 | 1.00 | 1.00 | 0.9948 | 1.00 | 1.00 | 0.9956 | 0.9816 | 0.7081 | 0.4591 | 0.9035 |
| 1dConv test acc| 0.5243 | 0.7028 | 0.9397 | 0.995 | 1 | 1 | 0.9948 | 1 | 1 | 0.9956 | 0.9816 | 0.7081 | 0.4591 | 0.9035 |
| LSTM test acc| 0.5243 | 0.7028 | 0.9397 | 0.995 | 1 | 1 | 0.9948 | 1 | 1 | 0.9956 | 0.9816 | 0.7081 | 0.4591 | 0.9035 |
| Mix test acc | 0.5243 | 0.7028 | 0.9397 | 0.995 | 1 | 1 | 0.9948 | 1 | 1 | 0.9956 | 0.9816 | 0.7081 | 0.4591 | 0.9035 |
