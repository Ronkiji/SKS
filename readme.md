# SKS

This repository provides code for the paper "Hate Speech Detection based on Sentiment Knowledge Sharing" with adjustments made by Aryan Chawla and Ronald Sun.

![avatar](figure1.jpg)

# Requirements

They are illustrated in the `requirements.txt` file. However, it is reccomended to install the dependencies one-by-one rather then with `-r` option. If pyenchant errors occur, please use `pyenchant==3.1.0`. Finally, you may need to interface with python and install `punkt` if it is the first time using `nltk` and/or the project.

# Data

[DV](https://github.com/t-davidson/hate-speech-and-offensive-language), [SE](https://github.com/rnjtsh/hatEval-2019/blob/master/public_development_en/dev_en.tsv), [SA](https://www.kaggle.com/dv1453/twitter-sentiment-analysis-analytics-vidya)

The SE dataset may need some adjustment in formatting from tsv to csv. Make sure to put these in the data directory and also within their respective directory too. ex: `SemEval_task5/df_test.csv`

The glove txt file can be downloaded [here](https://www.kaggle.com/datasets/aellatif/glove6b300dtxt). There is also a larger one available, but make sure to adjust the script for it [here](https://www.kaggle.com/datasets/authman/pickled-glove840b300d-for-10sec-loading)

# Usage

After download the data and the pre-trained word vectors, just run the bash script associated with the datset.
