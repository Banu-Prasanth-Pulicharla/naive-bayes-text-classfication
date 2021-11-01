# Naive Bayes for Text Classification
## Description
Implemented and evaluate Naive Bayes for text classification.

Download the spam/ham (ham is not spam) dataset available on repo. The data set is divided into two sets: training set and test set. The dataset was used in the Metsis et al. paper [1]. Each set has two directories: spam and ham. All files in the spam folders are spam messages and all files in the ham folder are legitimate (non spam) messages.

Implemented the multinomial Naive Bayes algorithm for text classification described here: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf (*see Figure 13.2*).

Note that the algorithm uses add-one laplace smoothing. Ignored punctuation and special characters and normalized words by converting them to lower case, converting plural words to singular (i.e., `Here` and `here` are the same word, `pens` and `pen` are the same word). Normalized words by stemming them using an online stemmer such as http://www.nltk.org/howto/stem.html.

> All the calculations are in log-scale to avoid underow.

Later, Improved the Naive Bayes by throwing away (i.e., filtering out) stop words such as `the` `of` and `for` from all the documents. A list of stop words can be found here: http://www.ranks.nl/stopwords.

## How to Run?
a. Place the file `NaiveTextClassification.py` in a directory.  
b. use below command to run the script -   
   ```
   python TextClassification.py
   ```
c. Parameters for the script would be asked now. Please provide in below format -   
   ```
   <Training Set Ham Path> <Training Set Spam Path> <Test Set Ham Path> <Test Set Spam Path>
   ```
   Ex:-   
   ```
   D:\data_TEMP\train\ham D:\data_TEMP\train\spam D:\data_TEMP\test\ham D:\data_TEMP\test\spam
   ```
d. That's it! Output would show the accuracies for training, test data with and without stopwords.