__author__ = 'Guillaume'

import cv2 # OpenCV
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os

def remove_black_borders_from_rotation(rotated_img, angle):

    if angle<0:
        out = remove_black_borders_from_rotation(rotated_img[:,::-1], -angle)
        return out[:,::-1]

    row, col = rotated_img.shape

    first_row = rotated_img[0]
    first_col = rotated_img[:,0]

    cropx = np.argwhere(np.cumsum(first_row[::-1]!=0)==1)[0][0]
    cropy = np.argwhere(np.cumsum(first_col!=0)==1)[0][0]

    return rotated_img[cropy:(row-cropy), cropx:(col-cropx)]

def remove_black_borders_from_translation(translated_img, tx, ty):

    row, col = translated_img.shape

    left = max(0,tx)
    right = min(col,col+tx)
    up = max(0,ty)
    down = min(row,row+ty)

    return translated_img[up:down, left:right]

def rotate(img, angle):
    pil_img = PIL.Image.fromarray(img)
    return np.array(pil_img.rotate(angle), dtype=img.dtype)

def resize(img, size, interpolation = cv2.INTER_CUBIC):
    return cv2.resize(img, size, interpolation = interpolation)

def resize_and_scale(img, size, scale, interpolation = cv2.INTER_CUBIC):
    img = resize(img, size, interpolation)
    return np.array(img, "float32")/scale

def translate(img, tx, ty):
    if img.ndim > 2:
        print "Rotate only works for gray image."
    rows,cols = img.shape
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(img,M,(cols,rows))

def add_channel(img):
    return img[None,:,:]

def convert_to_grayscale(img):
    #return PIL.Image.fromarray(img).convert("L")
    return cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

def crop(img, crop_rates):
    rows,cols = img.shape
    crop_up, crop_down, crop_left, crop_right = crop_rates
    crop_up = int(crop_up*rows/100.0)
    crop_down = int(crop_down*rows/100.0)
    crop_left = int(crop_left*rows/100.0)
    crop_right = int(crop_right*rows/100.0)
    return img[crop_up:(rows-crop_down),crop_left:(cols-crop_right)]

def rotate_crop_and_scale(img, final_size, max_angle, max_crop_rate, scale):
    # Random Rotation
    angle = np.random.randint(-max_angle,max_angle)
    rotated_img = rotate(img, angle)
    # Remove black borders
    cropped_rotated_img = remove_black_borders_from_rotation(rotated_img, angle)
    # Random Crop
    crop_rates = np.random.randint(0, max_crop_rate, 4) # translation rates
    cropped_im = crop(cropped_rotated_img, crop_rates)
    # Resize and scale
    try:
        resized_img = resize_and_scale(cropped_im, final_size, scale, interpolation=cv2.INTER_CUBIC)
    except: # sometimes i get an error when resizing with open cv (ssize.area < 0 error)
        resized_img = np.zeros(final_size, dtype="float32")
    # Flip left right
    if np.random.randint(0,2)==1:
        resized_img = np.fliplr(resized_img)
    output_img = resized_img[None,:,:]
    # Return
    return output_img

def convert_labels(labels_1D):
    """
    Convert a binary 1D label into a 2D one : 0->[0,1], 1->[1,0]
    :param labels_1D: [1,0,1,0,0...]
    :return: [[1,0],[0,1]...]
    """
    targets = np.zeros((len(labels_1D),2))
    targets[:,0] = labels_1D.flatten()
    targets[:,1] = 1-labels_1D.flatten()
    return np.array(targets, "float32")

def generator_from_file(file, file_target, batch_size, preprocessing_func, preprocessing_args):
    """
    Generator function used when using the keras function 'fit_on_generator'. Load the trainset as a numpy
    array, and the corresponding targets, and returns a tuple to the training containing a processed batch and
    targets. This can be done on the CPU, in parallel of a GPU training. See 'fit_on_generator' for more details.

    :param file: path to training data
    :param file_target: path to training targets
    :param batch_size:
    :param preprocessing_func: function which will be applied to each training batch
    :param preprocessing_args: arguments of the preprocessing function
    :return: tuple(batch,targets)
    """
    while 1:
        trainset = np.load(file)
        targets = np.load(file_target)
        for i in range(trainset.shape[0]/batch_size)[0:-1]:
            batch = np.zeros((batch_size, 1, 100, 100), dtype="float32")
            for k in range(batch_size):
                batch[k,0] = preprocessing_func(trainset[i*batch_size+k,0], *preprocessing_args)
            yield batch, targets[i*batch_size:((i+1)*batch_size)]
