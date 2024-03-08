# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
#Import for command line functions
import argparse

#Class imports
#Import functions for preprocessing
from preprocessing_data import *
#Import functions for feature selection
from feature_extraction import *
#Import functions for naiveBayes classification
from multinomial_naive_bayes import *
#Import functions for evaluation of predictions
from evaluation import *
from file_handling_operations import *
from train_naive_bayes import *
from sentence import *


USER_ID = "aca20sf" 


def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

def apply_feature_selection(words, features, number_classes):
    """
    Applying feature selection if feature selection selected.
    
    Args:
        words (list):     Sentence objects list.
        features (str): 'all_words' or 'features'.
        number_classes (int): Number of classes for sentiment classification, 3 or 5.
    """
    if features == "features":
        featureProcessor = FeatureExtraction(number_classes)
        featureProcessor.featureExtraction(words, number_classes)

def save_prediction_results(output_files, number_classes, predicted_dev_sentiments, dev_words, predicted_test_sentiments, test_words):
    """
    Save prediction results to files if the output files selected.
    
    Args:
        output_files (bool): Flag to indicate if prediction files need to be saved.
        number_classes (int): Number of classes for sentiment classification, 3 or 5.
        predicted_dev_sentiments (list): Predicted sentiment list for dev dataset.
        dev_words (list): Sentence objects list for dev dataset.
        predicted_test_sentiments (list): Predicted sentiments list for test dataset.
        test_words (list): Sentence objects list for test dataset.
        
    Returns:
        tuple: A tuple of paths to the saved dev and test prediction files.
    """
    dev_file = f"dev_predictions_{number_classes}classes_{USER_ID}.tsv"
    test_file = f"test_predictions_{number_classes}classes_{USER_ID}.tsv"

    if output_files:
        # Save predictions for dev dataset
        FileHandlingOperations.save_predictions_to_file(dev_file, predicted_dev_sentiments, dev_words)
        # Save predictions for test dataset
        FileHandlingOperations.save_predictions_to_file(test_file, predicted_test_sentiments, test_words)

    return dev_file, test_file

def handle_predictions(output_files, number_classes, predicted_dev_sentiments, dev_words, predicted_test_sentiments, test_words):
    """
    Stores and retrieves prediction results and evaluates F1 score.
    
    Args:
        output_files (bool): Flag to indicate if prediction files need to be saved.
        number_classes (int): Number of classes for sentiment classification, 3 or 5.
        predicted_dev_sentiments (list): Predicted sentiment list for dev dataset.
        dev_words (list): Sentence objects list for dev dataset.
        predicted_test_sentiments (list): Predicted sentiments list for test dataset.
        test_words (list): Sentence objects list for test dataset.
        
    Returns:
        tuple: A tuple of the calculated macro F1 score for the dev dataset and the 
               Evaluation object with calculated metrics.
    """
    # Save the prediction results if the flag is set
    dev_file, test_file = save_prediction_results(output_files, number_classes, predicted_dev_sentiments, dev_words, predicted_test_sentiments, test_words)

     # Evaluate the prediction results using the Evaluation class
    predictions_dict = {word.sentenceID: label for word, label in zip(dev_words, predicted_dev_sentiments)}
    evaluation = Evaluation(dev_words, predictions_dict, number_classes)
    f1_score = evaluation.macro_f1_score()

    return f1_score, evaluation

def main():
    inputs=parse_args()
   
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
   
    #Load training dataset
    train_words = Sentence.wordList(training)
    
    sentiment_scale = [0,1,2,3,4]
    if number_classes == 3:
        #Preprocess training data
        preprocess = PreprocessingData(train_words)
        preprocess.preprocess_words()
        # Rescale training data
        sentiment_scale = [0,1,2]
        preprocess.scale_down_sentiment()

    #Feature selection on training data
    apply_feature_selection(train_words, features, number_classes)
        
    #Train the naive bayes classifier
    data = TrainNaiveBayes(train_words, sentiment_scale)

    #Load dev dataset
    dev_words = Sentence.wordList(dev)
    #Load test dataset
    test_words = Sentence.wordList(test)

    sentiment_scale = [0,1,2,3,4]
    if number_classes == 3:
        #Preprocess dev data
        preprocess_dev = PreprocessingData(dev_words)
        preprocess_dev.preprocess_words()
        #Preprocess test data
        preprocess_test = PreprocessingData(test_words)
        preprocess.preprocess_words()
        sentiment_scale = [0,1,2]
        # Rescale dev data
        preprocess_dev.scale_down_sentiment()
        # Rescale test data
        preprocess_test.scale_down_sentiment()
        
    #Feature selection on dev data
    apply_feature_selection(dev_words, features, number_classes)
    #Feature selection on test data
    apply_feature_selection(test_words, features, number_classes)

    predict_sentiments = MultinomialNaiveBayes(data)
    # Predicting dev sentiments
    predicted_dev_sentiments = predict_sentiments.sentiments(dev_words)
    # Predicting test sentiments
    predicted_test_sentiments = predict_sentiments.sentiments(test_words)
    
    # Making predictions on the development and test datasets and writing to prediction files    
    f1_score, evaluation = handle_predictions(output_files, number_classes, predicted_dev_sentiments,
                                              dev_words, predicted_test_sentiments, test_words)

    if confusion_matrix:
        # Plot confusion matrix
        evaluation.plot_confusion_matrix()

        
   # # Error analysis code
   #  actual_dev_sentiments = FileHandlingOperations.load_actual_sentiments_from_file(dev)
   #  predicted_dev_sentiments = FileHandlingOperations.load_prediction_from_file(dev_file)
   #  misclassified_instances = []
    
   #  for sentence_id in actual_dev_sentiments:
   #      actual_sentiment = actual_dev_sentiments[sentence_id]
   #      predicted_sentiment = predicted_dev_sentiments.get(sentence_id)
    
   #      if (actual_sentiment == 0 and predicted_sentiment == 2) or (actual_sentiment == 2 and predicted_sentiment == 0):
   #          misclassified_instances.append((sentence_id, actual_sentiment, predicted_sentiment))
    
   #  print("Misclassified Instances:")
   #  for sentence_id, actual, predicted in misclassified_instances:
   #      print(f"Sentence ID: {sentence_id}, Actual Sentiment: {actual}, Predicted Sentiment: {predicted}")

    

    #Print the results and relevant console information
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()