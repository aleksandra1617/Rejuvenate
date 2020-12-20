# TODO:
#######################################################################################################################
#                                             Data Mining Algorithms                                                  #
#                                                                                                                     #
#           This project was developed for the Machine Learning module at Teesside University with the aim            #
#           to demonstrate and evaluate the use of popular computational techniques for machine learning.             #
#           The project contains implementations of popular techniques for Image Analysis such as Support             #
#           Vector Machine, Convolutional Neural Network and Hidden Markov Models.                                    #
#                                                                                                                     #
#                                                                                                                     #
#   Data Repository Contains:                                                                                         #
#       > PlantVillage Dataset                                                                                        #
#                                                                                                                     #
#                                                  Developed by                                                       #
#                                               Aleksandra Petkova                                                    #
#                                                                                                                     #
#######################################################################################################################

from os import getcwd
import sys
import numpy as np
import ImageProcessing
import CNN
import NBN
import SVM

# Dataset Constants
DATASET_PATH = getcwd() + "\\Data Repository\\PlantVillage\\Dataset"
WIDTH, HEIGHT, DEPTH = 256, 256, 3
DEFAULT_IMG_SIZE = tuple((WIDTH, HEIGHT))


# Run Convolutional Neural Network (Core Python)
def run_cnn_core(argv):
    print("\nSTARTING CONVOLUTIONAL NEURAL NETWORK (CORE PYTHON VERSION)..")


# Run Convolutional Neural Network (Library)
def run_cnn_lib(argv):
    print("\nSTARTING CONVOLUTIONAL NEURAL NETWORK (LIBRARY VERSION)..")
    CNN.run(argv)


# Run Hidden Markov Models (Core Python)
def run_hmm_lib(argv):
    print("\nSTARTING HIDDEN MARKOV MODELS (CORE PYTHON VERSION)..")


# Run Support Vector Machine (Library)
def run_svm_lib(argv):
    print("\nSTARTING SUPPORT VECTOR MACHINE (LIBRARY VERSION)..")


def main(argv):
    # Datasets contains all the available datasets in this order:
    # 0: original_data, 1: denoise_original_data, 2: sample_original_data, 3: denoised_sample_data
    print("Loading 500 images..")
    datasets = np.array(ImageProcessing.run(500, 5, getcwd() + "\\Data Repository\\PlantVillage\\Dataset"))

    alg_run_dict = {'1': run_cnn_lib, '2': run_hmm_lib, '3': run_svm_lib}
    #print('argv : ', argv[0])
    algorithm_selection = None

    while algorithm_selection != 0:
        print("\n===================== Main Menu ====================="
           "\nPlease select the algorithm you wish to run! (1 to 3)"
          "\n\t0) Exit"
          "\n\t1) Convolutional Neural Network (Library)"
          "\n\t2) Hidden Markov Models (Library)"
          "\n\t3) Support Vector Machine (Library)")

        algorithm_selection = input("\nEnter chosen option here: ")

        while not (algorithm_selection.isdigit() and 0 < int(algorithm_selection) < 4):
            if algorithm_selection == '0':
                print("Closing program..")
                return 0

            print("\nInvalid input, must be a number between 0 and 3, inclusive.")
            algorithm_selection = input("Re-enter option here: ")
        else:
            alg_run_dict[algorithm_selection](datasets[0])
            print("Returning to main menu..")


if __name__ == '__main__':
    main(sys.argv[1:])
