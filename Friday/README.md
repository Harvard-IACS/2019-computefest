### Harvard ComputeFest 2019

#### Actionable Understanding of Customer Reviews via Classification and Summarization

 - Pavlos Protopapas, WeiWei Pan, Spandan Madan, Srivatsan Srinivasan
 
#### INSTRUCTIONS

##### PART 1 : CLASSIFICATION

1. Install gensim package.https://pypi.org/project/gensim/

##### PART 2 : SUMMARIZATION
2. Ensure that you have the following packages available in your python environment - pandas, gensim, numpy, scipy, nltk, sklearn, keras, torch, logging, re, collections, pickle, os, sys, random, tensorflow.

3. Please download and install GLoVE embeddings - http://nlp.stanford.edu/data/glove.6B.zip. Particularly unzip the 100B file into your local directory by creating a folder called embeddings and then datasets inside that and placing it there. Effectively it will be the base (should be \2019-computefest\Friday) directory\embeddings\dataset\glove.6B.100d.txt

4. For the purposes of this workshop, we will use only the dataset given in this repo and not the full dataset since we are not going to train the models but rather just focus on preprocessing. The complete dataset for reviews can be found here - https://www.kaggle.com/snap/amazon-fine-food-reviews?fbclid=IwAR1IDqdxTDLMFOX1dZxd26sUpeoAx0XOIMDupaP_DMRekdr4yRsFZP6Q2AM . If you are training the model on your own machine, please use this dataset in its whole and train for reasonable results.

###### We also thank the set of repositories [1](https://github.com/llSourcell/How_to_make_a_text_summarizer),[2](https://github.com/DeepsMoseli/Bidirectiona-LSTM-for-text-summarization) that were very useful while making this material -  


We will try to publish some saved models for inference after the workshop. Thank you!
