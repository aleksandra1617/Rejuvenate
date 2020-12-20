from os import getcwd, listdir
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from matplotlib import pyplot as plt
from Utilities import threadpool

WIDTH, HEIGHT, DEPTH = 980, 720, 3
DEFAULT_IMG_SIZE = tuple((WIDTH, HEIGHT))

PROCESSES = 0


def get_image_paths(dataset_path, images_paths_map):
    """
      Uses the os module to find all the images from the dataset and load them into the program.
      Before running this function the dataset needs to be modified to contain a directory per disease
      nested into a directory per plant variety like so - 'Bell Pepper\\Bacterial Spot', 'Bell Pepper\\Healthy'.

      IMPORTANT NOTE: THE SEARCH RELIES ON FINDING A '.' SYMBOL TO DECIDE IF THE ITEM DETECTED IS A FILE OR A DIRECTORY
      TO NOT CONFUSE THE ALGORITHM PLEASE MAKE SURE THERE ARE NO . IN THE NAMES OF THE DIRECTORIES.

      Parameters
      ----------
      (string) dataset_path: the path from which the image scan should start.
      (dictionary) images_paths_list: this parameter should be a reference of an empty dictionary to store
      the labels and paths data for each detected image.

      Returns
      -------
      (list) image_list: 3D list containing a 2D list representation of an image, where the innermost list contains
      the B, G, R values of a pixel in an image.
      (list) label_list: the labels of the images in image_list, linked by index.
      """

    try:
        # Lists all the items in the directory with path 'dataset_path'.
        # This could be a list of files, list of directories or list of a combination of both.
        dataset_directory = listdir(dataset_path)

        # Check if the items in dataset directory are files or directories
        for name in dataset_directory:
            new_path = f"{dataset_path}\\{name}"

            # if it is an image, store it and continue through the rest of the items.
            if name.endswith(".jpg") or name.endswith(".JPG") or \
                    name.endswith(".PNG") or name.endswith(".png"):
                label = dataset_path.split("\\")[-1]  # Extract the Label from the path.

                # If this label exists in the dictionary append the new path
                if label in images_paths_map.keys():
                    images_paths_map[label].append(new_path)
                else:  # Otherwise create the label, its list of paths and append.
                    images_paths_map.update({label: [new_path]})

            elif '.' not in name:  # if it is a directory, spawn a recursion to keep searching.
                get_image_paths(new_path, images_paths_map)

    except Exception as e:
        print(f"[Error] : {e}")


def denoise_image(original_image):
    # Denoising original image
    dst = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 21)  # Source of slow processing.

    # Reorder the BGR values of the denoised image to RGB so that it can be loaded correctly
    # b, g, r = cv2.split(dst)
    # rgb_denoised = cv2.merge([r, g, b])

    # Reorder the BGR values of the original image to RGB so that it can be loaded correctly
    # b, g, r = cv2.split(original_image)
    # rgb_original = cv2.merge([r, g, b])

    # plt.subplot(211), plt.imshow(rgb_original)
    # plt.subplot(212), plt.imshow(rgb_denoised)
    # plt.show()

    return dst


def format_image(image_path):
    """
    Loads an images as a 2D list of BGR pixel data, re-formats it and returns the processed image.

    Parameters
    ----------
    (string) image_path: the path to an image.
    (tuple of int) img_size: should contain a tuple structure to pass into the cv2.resize function.
    Format -> tuple((WIDTH, HEIGHT))

     Returns
     -------
     (list) original_image: 3D list containing a 2D list representation of an image, where the innermost list contains
     the B, G, R values of a pixel in an image.
     (list) denoised_image: 3D list containing a 2D list representation of an image, where the innermost list contains
     the B, G, R values of a pixel in an image.
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            original_image = img_to_array(image)

            denoised_image = img_to_array(denoise_image(image))  # cv2.resize(image, image_size)
            return original_image, denoised_image
        else:
            print(f"[Warning] : Unable to read the image, returning empty array..")
            return np.array([]), np.array([])

    except Exception as e:
        print(f"[Error] : {e}")
        return None, None


@threadpool
def process_image(image_paths, key, original_dataset, denoised_original_dataset, sample_size, sample_sample_size, sample_dataset, denoised_sample_dataset,
                  original_dataset_out, denoised_original_dataset_out, sample_dataset_out, denoised_sample_dataset_out):

    global PROCESSES

    temp_original, temp_denoised = [], []

    PROCESSES += 1

    for i in range(len(image_paths[key][:sample_size])):
        path = image_paths[key][i]
        original, denoised = format_image(path)
        if original is not None: temp_original.append(original)
        if denoised is not None: temp_denoised.append(denoised)

        # Clone images from each category as described by sample_size to generate sample data.
        if len(temp_original) == sample_sample_size: sample_dataset += temp_original
        if len(temp_denoised) == sample_sample_size: denoised_sample_dataset += temp_denoised

        original_dataset_out.append(key)
        denoised_original_dataset_out.append(key)
        if len(temp_original) <= sample_sample_size: sample_dataset_out.append(key)
        if len(temp_original) <= sample_sample_size: denoised_sample_dataset_out.append(key)

        print("[INFO] Finished Processing", i, " Images from  << ", key, " >>  category..")

        # Merge the accumulated data
    original_dataset += temp_original
    denoised_original_dataset += temp_denoised
    PROCESSES -= 1
    print("[INFO] Finished Processing Images from  << ", key, " >>  category..")


def run(sample_size, sample_sample_size, root_dir):
    global PROCESSES
    image_paths = {}
    get_image_paths(root_dir, image_paths)

    # Load and format all the images.
    original_dataset, denoised_original_dataset, sample_dataset, denoised_sample_dataset = [], [], [], []
    original_dataset_out, denoised_original_dataset_out, sample_dataset_out, denoised_sample_dataset_out = [], [], [], []

    for key in image_paths:
        print("[INFO] Processing Images from  << ", key, " >>  category..")

        process_image(image_paths, key, original_dataset, denoised_original_dataset, sample_size, sample_sample_size,
                      sample_dataset, denoised_sample_dataset, original_dataset_out, denoised_original_dataset_out,
                      sample_dataset_out, denoised_sample_dataset_out)

    while PROCESSES != 0:
        pass

    original_data = [original_dataset, original_dataset_out]
    denoise_original_data = [denoised_original_dataset, denoised_original_dataset_out]
    sample_original_data = [sample_dataset, sample_dataset_out]
    denoised_sample_data = [denoised_sample_dataset, denoised_sample_dataset_out]

    return original_data, denoise_original_data, sample_original_data, denoised_sample_data


if __name__ == '__main__':
    run(3, 2, getcwd() + "\\Data Repository\\PlantVillage\\Dataset")
