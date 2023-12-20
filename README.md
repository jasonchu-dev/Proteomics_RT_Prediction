# Proteomics Retention Time Prediction

Dataset: https://www.kaggle.com/datasets/kirillpe/proteomics-retention-time-prediction

## peptide2RT

### Encoder

* Embedding
* Dropout
* 2 LSTM layers

### Decoder

* 1d convolutional layer
* Batchnorm
* Sigmoid
* 3 layers each