__author__ = 'Guillaume'

#import cv2 # OpenCV
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
from dataset import InMemoryDataset, FuelDataset

def remove_black_borders_from_rotation(rotated_img, angle):

    if angle<0:
        out = remove_black_borders_from_rotation(rotated_img[:,::-1], -angle)
        return out[:,::-1]

    row, col = rotated_img.shape[0:2]

    first_row = rotated_img[0]
    first_col = rotated_img[:,0]

    try:
        cropx = np.argwhere(np.cumsum(first_row[::-1]!=0)==1)[0][0]
        cropy = np.argwhere(np.cumsum(first_col!=0)==1)[0][0]
    except:
        cropx=0.2*row
        cropy=0.2*col

    if cropy>(0.4*row) or cropx>(0.4*col):
        return rotated_img

    return rotated_img[cropy:(row-cropy), cropx:(col-cropx)]

def remove_black_borders_from_translation(translated_img, tx, ty):

    row, col = translated_img.shape[0:2]

    left = max(0,tx)
    right = min(col,col+tx)
    up = max(0,ty)
    down = min(row,row+ty)

    return translated_img[up:down, left:right]

def rotate(img, angle):
    if img.ndim == 3 and img.shape[2]==1:
        pil_img = PIL.Image.fromarray(img[:,:,0])
    else:
        pil_img = PIL.Image.fromarray(img)
    return np.array(pil_img.rotate(angle), dtype=img.dtype).reshape(img.shape)

def resize_pil(img, size, interpolation = PIL.Image.BICUBIC):
    if img.ndim == 3 and img.shape[2]==1:
        pil_img = PIL.Image.fromarray(img[:,:,0])
        return np.array(pil_img.resize(size, interpolation), dtype=img.dtype)[:,:,None]
    else:
        pil_img = PIL.Image.fromarray(img)
        return np.array(pil_img.resize(size, interpolation), dtype=img.dtype)

# def resize_cv2(img, size, interpolation = cv2.INTER_CUBIC):
#     if img.ndim == 2:
#         return cv2.resize(img, size, interpolation = interpolation)
#     else:
#         if img.shape[2] == 3:
#             return cv2.resize(img, size, interpolation = interpolation)
#         else:
#             return cv2.resize(img, size, interpolation = interpolation)[:,:,None]

def resize_and_scale(img, size, scale, interpolation = PIL.Image.BICUBIC):
    img = resize_pil(img, size, interpolation)
    return np.array(img, "float32")/scale

# def translate(img, tx, ty):
#     if img.ndim > 2:
#         print "Rotate only works for gray image."
#     rows,cols = img.shape
#     M = np.float32([[1,0,tx],[0,1,ty]])
#     return cv2.warpAffine(img,M,(cols,rows))

def add_channel(img):
    return img[None,:,:]

def convert_to_grayscale(img):
    return PIL.Image.fromarray(img).convert("L")

def crop(img, crop_rates):
    rows,cols = img.shape[0:2]
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
    resized_img = resize_and_scale(cropped_im, final_size, scale, interpolation=PIL.Image.BICUBIC)
    # Flip left right
    if np.random.randint(0,2)==1:
        resized_img = np.fliplr(resized_img)
    #output_img = resized_img[None,:,:]
    # Return
    return resized_img

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

def images_generator(data_access, dataset, targets, batch_size, tmp_size, final_size, bagging_size, bagging_iterator,
                     preprocessing_func, preprocessing_args):
    """
    Generator function used when using the keras function 'fit_on_generator'. Can work with InMemoryDataset, FuelDataset.
    Yield a tuple to the training containing a processed batch and
    targets. This can be done on the CPU, in parallel of a GPU training. See 'fit_on_generator' for more details.

    :param data_access: "in-memory" or "fuel"
    :param dataset: path to the dataset numpy file (not used when data_acces = "fuel")
    :param targets: path to the targets numpy file (not used when data_acces = "fuel")
    :param batch_size:
    :param tmp_size: Used when data_access == "fuel". Datastream will return images of size equal to tmp_size.
    :param final_size: size of images used for the training
    :param preprocessing_func: function which will be applied to each training batch
    :param preprocessing_args: arguments of the preprocessing function
    :return: tuple(batch,targets)
    """
    if data_access=="in-memory":
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets)
    elif data_access=="fuel":
        train_dataset = FuelDataset("train", tmp_size, batch_size=batch_size, bagging=bagging_size,
                                    bagging_iterator=bagging_iterator)
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)
    while 1:
        # Get next batch
        batch,batch_targets = train_dataset.get_batch()
        # Pre-processing
        processed_batch = np.zeros((batch.shape[0],final_size[2],final_size[0],final_size[1]),
                                   dtype="float32")
        for k in range(batch_size):
            processed_batch[k] = preprocessing_func(batch[k], *preprocessing_args).transpose(2,0,1)
        # Send Batch
        labels = convert_labels(batch_targets)
        yield processed_batch,labels


def check_preprocessed_data(data_access, dataset, targets, batch_size, tmp_size, final_size, preprocessing_func,
                            preprocessing_args, n=10):
    if data_access=="in-memory":
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets)
    elif data_access=="fuel":
        train_dataset = FuelDataset("train", tmp_size, batch_size=batch_size)
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)

    # Compute only one batch
    start=time.time()
    batch,batch_targets = train_dataset.get_batch()
    batch_targets = convert_labels(batch_targets)
    processed_batch = np.zeros((batch.shape[0],final_size[2],final_size[0],final_size[1]),
                                   dtype="float32")
    for k in range(batch_size):
        processed_batch[k] = preprocessing_func(batch[k], *preprocessing_args).transpose(2,0,1)
    end=time.time()

    print "Batch Shape = ", processed_batch.shape, "with dtype =", processed_batch.dtype
    print "Targets Shape =", batch_targets.shape, "with dtype =", batch_targets.dtype
    for i in range(n):
        plt.figure(0)
        plt.gray()
        plt.clf()
        plt.title("(%d,%d)"%(batch_targets[i][0], batch_targets[i][1]))
        if batch.shape[1]==3:
            plt.imshow(processed_batch[i].transpose(1,2,0))
        else:
            plt.imshow(processed_batch[i,0])
        plt.show()
    print "Processing 1 batch took : %.5f"%(end-start)
