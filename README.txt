# COM3110: Sentiment Analysis of Movie Reviews

Implementation of a multinomial Naive Bayes classifier for sentiment analysis using the Rotten Tomatoes movie review dataset.

The project has the following files:
- NB_sentiment_analyser.py - The main file to run the sentiment analysis.
- preprocessing_data.py - Preprocesses the data.
- feature_extraction.py - Implements feature extraction.
- multinomial_naive_bayes.py - Contains the Multinomial Naive Bayes classifier.
- evaluation.py - Evaluates the model.
- file_handling_operations.py - Provides functionality for file operations.
- train_naive_bayes.py - Trains the model.
- sentence.py - Represents a sentence in the dataset.
- sentiment_bias.py - Implements sentiment bias calculation.
- sentiment_lexicon.py - Implements sentiment lexicon feature extraction.
- tfidf.py - Implements the TF-IDF calculations.

## Evaluation Results

The models' performance is evaluated using the macro-average F1 score. The macro-f1 score provides a balance between the model's precision and recall across all classes. Moreover, confusion matrices are used to visualize the performance of the model in classifying sentiments correctly. These metrics help understand the model's strengths and weaknesses, allowing for targeted improvements.

## Setting up the programme

 - Make sure you have Python 3.9.x or above installed, along with the following libraries: numpy, pandas, seaborn, matplotlib, and nltk. The requirements.txt file lists the dependencies and their versions.

- Install the NLTK library by executing the following commands in the Anaconda prompt:

     pip install nltk
     import nltk
     nltk.download('all')

## Running the programme

- Navigate to the correct directory and execute the following command:

python NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes <3 or 5> -features <all_words or features> -confusion_matrix

where:
• <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and
test files, respectively;
• -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being
predicted;
• -features is a parameter to define whether you are using your selected features or
no features (i.e. all words);
• -output_files is an optional value defining whether or not the prediction files should
be saved (see below – default is "files are not saved"); and
• -confusion_matrix is an optional value defining whether confusion matrices should
be shown (default is "confusion matrices are not shown").
