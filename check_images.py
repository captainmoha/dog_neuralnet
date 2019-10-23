#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# Fill in your information in the programming header below
# PROGRAMMER: Mohamed A. Farouk
# DATE CREATED: 22/08/2019
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir, mkdir
from os.path import exists, isfile
from random import randint
# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports for using PIL to add classification labels to images
from PIL import Image, ImageDraw, ImageFont


# Main program function defined below
def main():
    # 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # get command line arguments
    in_arg = get_input_args()
    # check_command_line_arguments(in_arg)

    # create pet image labels by creating a dictionary with key=filename and value=file label
    #  to be used to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # TODO: 4. Define classify_images() function to create the classifier
    # labels with the classifier function using in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    label_images(result_dic, in_arg.dir)

    check_classifying_images(result_dic)
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog()

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats()

    # TODO: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results()

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_seconds = end_time - start_time
    hours = round(tot_seconds / 3600)
    mins = round((tot_seconds % 3600) / 60)
    secs = round((tot_seconds % 3600) % 60)

    tot_time = "{}:{}:{}".format(hours, mins, secs)
    print("\n** Total Elapsed Runtime:", tot_time)


# TODO: 2.-to-7. Define all the function below. Notice that the input
# parameters and return values have been left in the function's docstrings.
# This is to provide guidance for achieving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to achieve the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(
        description="Check images using a certain CNN model")
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help="Path to images files directory")
    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use for image classification(default - pick any of the following vgg, alexnet, resnet)')
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='Text file that contains all labels associated to dogs(default -"dognames.txt")')

    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these labels as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """

    petlabels_dic = {}
    filenames_list = listdir(image_dir)
    # print(len(filenames_list))

    for filename in filenames_list:
        if not isfile('{}/{}'.format(image_dir, filename)):
            continue
        label = filename[0:-4].lower()  # remove file extension
        label = " ".join(label.split("_")[:-1])    # remove digits from name & make label str 
        petlabels_dic[filename] = label
    
    # print(petlabels_dic)
    return petlabels_dic



def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """

    results_dic = {}

    for img_name, label in petlabel_dic.items():
        image_attrs = [label]
        img_classification = classifier(images_dir + img_name, model).lower()
        image_attrs.append(img_classification)
        
        
        image_attrs.append(check_match(img_classification, label))
        
        print('{:<50} is classified as: {:<40}'.format(
            img_name, img_classification))
        results_dic[img_name] = image_attrs

    return results_dic



def check_match(classification_str, label):
    classification_list = classification_str.split(', ')

    for classification in classification_list:

        # handle special case where classification has multiple words seperated by space
        has_space = classification.find(' ')
        if has_space != -1:
            multi_word_class = classification.split(' ')
            if label in multi_word_class:
                return 1
            
        if label in classification_list:
            return 1

    return 0
        

def adjust_results4_isadog():
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    pass


def calculates_results_stats():
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    pass


def print_results():
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """
    pass

def label_images(results_dic, img_dir):

    results_dir = img_dir + '/labeled_imgs'
    
    
    x, y = 30, 50 # text position
    color = 'rgb(255, 255, 255)'
    border_color = 'rgb(0, 0, 0)'
    
    if not exists(results_dir):
        mkdir(results_dir)

    # label images with classification
    print(len(results_dic))
    print("#################")
    for img_name, img_result in results_dic.items():
        
        # classification result label
        img_lbl = '\n'.join(img_result[1].title().split(', '))

        # open image to add label to it
        img = Image.open(img_dir + img_name)
        draw = ImageDraw.Draw(img)
        
        font_size = int(img.size[0] * .1) # adapt fontsize to image width

        font = ImageFont.truetype('fonts/Roboto-Bold.ttf', font_size)
        # draw border
        draw.text((x-1, y-1), img_lbl, font=font, fill=border_color)
        draw.text((x+1, y-1), img_lbl, font=font, fill=border_color)
        draw.text((x-1, y+1), img_lbl, font=font, fill=border_color)
        draw.text((x+1, y+1), img_lbl, font=font, fill=border_color)

        # draw  text
        draw.text((x, y), img_lbl, fill=color, font=font)
        img_path = '{}labeled_imgs/{}.jpg'.format(img_dir, img_result[1])

        # if another image has same classification, add a rand number suffix to its name
        if exists(img_path): 
            img.save('{}{}{}'.format(img_path[:-4], randint(1, 1000), img_path[-4:]))
        else:
            img.save(img_path)





# Call to main function to run the program
if __name__ == "__main__":
    main()
