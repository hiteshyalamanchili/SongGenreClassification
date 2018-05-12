# Classifying Song Genres Using Raw Lyric Data with Deep Learning
#### Collaborators: Connor Brennan, Sayan Paul, Hitesh Yalamanchili, Justin Yum (CS 194-129: Group 3)
#### University of California, Berkeley

With the advent of music streaming services in recent years and their respective recommendation algorithms, classifying song genres is a commercially useful application of deep learning. Our main focus was to use raw lyric data, rather than a bag of words or other types of summaries, so that we could capture positional information for each word. Specifically, we used lyric information to classify a song as one of $11$ genres. The three main architectures we investigated were the LSTM baseline, hierarchical multilayer  attention with GRUs (HAN-GRU), and Transformers with multihead attention. We found that stacking layers in the HAN-GRU was both the fastest to train and the best performer of the three, reinforcing the power of hierarchical attention. For a more in-depth look at our study, please take a look at our [report](finalreport.pdf).

## Getting Started

Before being able to use the models provided in `models/`, all you need to do is ensure you have the requirements to be able to train and test these models as well as obtain the data by unzipping the files located in `dataset`.

### Requirements
* Numpy
* Pandas
* Tensorflow
* Keras
* scikit-learn
* Tensor2tensor

### Obtaining the Pre-Processed Data

Unzip the two zip files `original_cleaned_lyrics.zip` and `english_cleaned_lyrics.zip`, which contain the pre-processed and cleaned song lyrics originally obtained from the MetroLyrics dataset and the English-only cleaned song lyrics respectively.
```
unzip original_cleaned_lyrics.zip
unizp english_cleaned_lyrics.zip
```

## Models
The various models that have been developed, trained, and tested extensively are linked below.

* [Vanilla LSTM Network](models/Vanilla%20LSTM%20Network.ipynb)
* [Bidirectional LSTM Network](models/Bidirectional%20LSTM%20Network.ipynb)
* [Transformer Network with Multihead Attention](models/Transformer%20Network%20with%20Multihead%20Attention.ipynb)
* **Final Model**: [Bidirectional GRU Network with Hierarchical Attention](models/HANGRU.ipynb)

## Acknowledgements
* Professor John Canny, University of California, Berkeley
* Some inspiration and implementation from https://github.com/Kyubyong/transformer/blob/master/modules.py
