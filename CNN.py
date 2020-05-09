# Utility Modules
from os import listdir, getcwd, path
import pickle
import numpy as np
import cv2

# Machine Learning Modules
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import ImageProcessing

from Utilities import time_function


# Dataset Constants
DATASET_PATH = getcwd() + "\\Data Repository\\PlantVillage\\Dataset"
SAMPLE_DATASET_PATH = getcwd() + "\\Data Repository\\PlantVillage\\SampleData"
WIDTH, HEIGHT, DEPTH = 256, 256, 3
DEFAULT_IMG_SIZE = tuple((WIDTH, HEIGHT))

# Training Constants
EPOCHS = 10
INIT_LR = 1e-3   # ???
BS = 32  # ???


# region IMAGE PRE-PROCESSING FUNCTIONS
# Function to convert images to array
def convert_image_to_array(image_path, img_size=DEFAULT_IMG_SIZE):
    """
    Loads an images as a 2D list of BGR pixel data.
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            #cv2.imshow("Loaded Image", image)
            #cv2.waitKey(0)
            return img_to_array(cv2.resize(image, img_size))
        else:
            print(f"[Warning] : Unable to read the image, returning empty array..")
            return np.array([])

    except Exception as e:
        print(f"[Error] : {e}")
        return None


def load_images():
    """
    Uses the os module to find all the images from the dataset and load them into the program.
    Before running this function the dataset needs to be modified to contain a directory per disease
    nested into a directory per plant variety like so - 'Bell Pepper\\Bacterial Spot', 'Bell Pepper\\Healthy'.

    Returns
    -------
    (list) image_list: 3D list containing a 2D list representation of an image, where the innermost list contains
    the B, G, R values of a pixel in an image.
    (list) label_list: the labels of the images in image_list, linked by index.
    """
    image_list, label_list = [], []
    dataset_path = DATASET_PATH
    try:
        print("[INFO] Loading images..")
        dataset_directory = listdir(dataset_path)

        # Navigate to each image to extract its path and load it into the program.
        for plant_directory in dataset_directory:
            plant_disease_list = listdir(f"{dataset_path}\\{plant_directory}")

            for plant_disease_name in plant_disease_list:
                print(f"[INFO] Processing {plant_disease_name}..")
                plant_disease_image_list = listdir(f"{dataset_path}\\{plant_directory}\\{plant_disease_name}")

                for image in plant_disease_image_list[:200]:
                    image_path = f"{dataset_path}\\{plant_directory}\\{plant_disease_name}\\{image}"

                    if image_path.endswith(".jpg") or image_path.endswith(".JPG"):
                        image_list.append(convert_image_to_array(image_path))
                        label_list.append(plant_disease_name)

        print("[INFO] Image Loading Complete.")

    except Exception as e:
        print(f"[Error] : {e}")

    return image_list, label_list
# endregion


@time_function
def run(dataset, chosen_model_file_name='cnn_model_epoch25.h5'):
    """
    Encloses the whole algorithm so that it may be run just by calling this function. Allows for cleaner main function
    and better separation of the files. By default the algorithm attempts to load the highest epoch cnn model available,
    which is cnn_model_epoch25.h5.
    TODO: Use Regex to automatically find all the available cnn models and select the highest epoch one to load as default.

    Parameters
    ----------
    (string) chosen_model_file_name: the name of the file containing the selected serialised cnn model.
    """
    # region DATA PRE-PROCESSING REGION

    # region Serialised Data Search Region
    # Serializable objects
    image_list, label_binarizer, image_labels, model = None, None, None, None

    # Searches for existing serialised data to load so that the data pre-processing phase can be skipped.
    serialised_data_found = path.isfile('image_data.pkl') \
                            and path.isfile('label_binarizer.pkl') \
                            and path.isfile('binarized_image_labels.pkl')

    data_prepared = 'N'
    if serialised_data_found:
        data_prepared = input("[INFO] Serialised image data found.. Would you like to deserialize the data? (Y/N)\n")
    # endregion

    if data_prepared.upper() == 'N':
        image_list, label_list = dataset[0], dataset[1]

        # Extract aLL the unique label values
        unique_labels = list(set(label_list))

        # Transform Image Labels uisng Scikit Learn's LabelBinarizer which applies the one-vs-all scheme
        # to allow multi-class classification and to gage how confident the network is in its answer.
        label_binarizer = LabelBinarizer()

        # Check the LabelBinarizer works as intended by applying the binarization to a list of unique labels.
        # This will provide a point of reference for the actual binarized image labels.
        test1_unique_labels = label_binarizer.fit_transform(unique_labels)
        test2_unique_labels = label_binarizer.fit_transform(unique_labels)

        # Apply the label binarisation to the actual label data
        image_labels = label_binarizer.fit_transform(label_list)

        # Serialise the label_binarizer object, the formatted_img_data and the binarized image labels.
        # This allows skipping all the steps before this line assuming the label_binarizer.pkl file exists.
        #pickle.dump(image_list, open('image_data.pkl', 'wb'))
        #pickle.dump(label_binarizer, open('label_binarizer.pkl', 'wb'))
        #pickle.dump(image_labels, open('binarized_image_labels.pkl', 'wb'))

    else:
        image_list = pickle.load(open("image_data.pkl", "rb"))
        label_binarizer = pickle.load(open("label_binarizer.pkl", "rb"))
        image_labels = pickle.load(open("binarized_image_labels.pkl", "rb"))

    num_classes = len(label_binarizer.classes_)
    print("\nList of Classes: ", label_binarizer.classes_)

    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    print("\n[INFO] Spliting data to train and test.")
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.3, random_state=42)
    # endregion

    # region TRAINING MODEL
    # If a serialisation of a trained cnn model exists we don't need to retrain
    # just load the file and run it with test data.
    if not path.isfile(chosen_model_file_name):
        # Start model construction here
        augmented = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

        model = Sequential()
        tensorShape = (HEIGHT, WIDTH, DEPTH)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            tensorShape = (DEPTH, HEIGHT, WIDTH)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=tensorShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        print(model.summary())

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        # distribution
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        print("\n [INFO] training network...")

        # Processing takes 20-30 min per Epoch TODO: improve processing speed.
        history = model.fit_generator(
            augmented.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
        )

        # Plot the train and val curve
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()
        plt.figure()

        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()

        # Save the model so that it can be reused
        print("[INFO] Saving model...")
        model_file_name = 'cnn_model_epoch'+str(EPOCHS)+'.pkl'
        # pickle.dump(model, open('cnn_model_epoch25.pkl', 'wb'))
        model.save('cnn_original_model_epoch10.h5')

    else:
        print("[INFO] Serialised data found.")
        print("[INFO] Loading Model..")
        model = models.load_model('cnn_model_epoch25.h5')  # chosen_model_file_name
    # endregion

    # Run the model with test data and asses accuracy.
    print("[INFO] Calculating model accuracy..")
    scores = model.evaluate(x=x_test, y=y_test)
    print(f"Test Accuracy: {scores[1] * 100}")


if __name__ == '__main__':
    datasets = np.array(ImageProcessing.run(500, 1, getcwd() + "\\Data Repository\\PlantVillage\\Dataset"))
    run(datasets)
    run(load_images())

