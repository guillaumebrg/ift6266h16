__author__ = 'Guillaume'

#import cv2 # OpenCV
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
from dataset import InMemoryDataset, FuelDataset
from scipy.ndimage.filters import gaussian_filter
from keras.models import model_from_json

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

def rotate_and_crop(img, max_angle, max_crop_rate):
    # Random Rotation
    angle = np.random.randint(-max_angle,max_angle)
    rotated_img = rotate(img, angle)
    # Remove black borders
    cropped_rotated_img = remove_black_borders_from_rotation(rotated_img, angle)
    # Random Crop
    crop_rates = np.random.randint(0, max_crop_rate, 4) # translation rates
    cropped_im = crop(cropped_rotated_img, crop_rates)
    return cropped_im

def rotate_crop_and_scale(img, final_size, max_angle, max_crop_rate, scale, blur=None,
                          rgb_alterate=False):
    cropped_im=rotate_and_crop(img, max_angle, max_crop_rate)
    # Resize and scale
    resized_img = resize_and_scale(cropped_im, final_size, scale, interpolation=PIL.Image.BICUBIC)
    # Flip left right
    if rgb_alterate:
        resized_img = rgb_alteration(resized_img)
    if blur is not None:
        resized_img = gaussian_filter(resized_img, blur)
    if np.random.randint(0,2)==1:
        resized_img = np.fliplr(resized_img)
    #output_img = resized_img[None,:,:]
    # Return
    return resized_img

def rotate_crop_and_standardize(img, final_size, max_angle, max_crop_rate, blur=None, eps=1e-3,
                                rgb_alterate=False):
    cropped_im=rotate_and_crop(img, max_angle, max_crop_rate)
    # Resize and scale
    resized_img = resize_pil(cropped_im, final_size, interpolation=PIL.Image.BICUBIC)
    if rgb_alterate:
        resized_img = rgb_alteration(resized_img)
    if blur is not None:
        resized_img = gaussian_filter(resized_img, blur)
    resized_img = (resized_img - resized_img.mean())/(resized_img.std()+eps)
    # Flip left right
    if np.random.randint(0,2)==1:
        resized_img = np.fliplr(resized_img)
    # Return
    return resized_img

def rotate_crop_and_mean(img, final_size, max_angle, max_crop_rate, blur=None,
                         rgb_alterate=False):
    cropped_im=rotate_and_crop(img, max_angle, max_crop_rate)
    # Resize and scale
    resized_img = resize_pil(cropped_im, final_size, interpolation=PIL.Image.BICUBIC)
    if rgb_alterate:
        resized_img = rgb_alteration(resized_img)
    if blur is not None:
        resized_img = gaussian_filter(resized_img, blur)
    resized_img = resized_img - resized_img.mean()
    # Flip left right
    if np.random.randint(0,2)==1:
        resized_img = np.fliplr(resized_img)
    # Return
    return resized_img

def preprocess_dataset(dataset, training_params, mode):
    if mode == "scale":
        dataset = dataset / training_params.scale
    if mode == "std":
        dataset = standardize_dataset(dataset, [1,2,3])
    if mode == "mean":
        dataset = remove_mean_dataset(dataset, [1,2,3])
    if mode == "blur":
        for i in range(dataset.shape[0]):
            dataset[i] = gaussian_filter(dataset[i], training_params.blur)

    return dataset

def rgb_alteration(im):
    rgb_components = np.array([[-0.57070609,-0.58842455,-0.57275746],
                               [ 0.72758705,-0.03900872,-0.68490539],
                               [ 0.38067261,-0.8076106,0.4503926 ]],"float32").T
    eigen_values = np.array([ 187.01200611,9.4903197,2.52593616 ], "float32")
    # Random alteration
    alpha = 0.1*np.random.randn(3)
    alpha *= eigen_values
    alterations = np.dot(alpha.reshape((1,3)), rgb_components)
    alterations = alterations.reshape((1,1,3))

    return im + alterations

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

def mean_with_list_axis(a, ax):
    out = np.mean(a, axis=ax[-1])
    if len(ax)>1:
        for i in ax[::-1][1:]:
            out = np.mean(out, axis=i)
    return out

def standardize_dataset(dataset, ax=[1,2,3], eps=1e-3):
    m = mean_with_list_axis(dataset, ax)
    shape = [1 for i in range(len(ax)+1)]
    shape[0]=dataset.shape[0]
    broadcast_m = m.reshape(shape)
    std = mean_with_list_axis(np.square(dataset-broadcast_m), [1,2,3])
    std = np.sqrt(std) + eps
    broadcast_std = std.reshape(shape)
    return (dataset-broadcast_m)/broadcast_std

def remove_mean_dataset(dataset, ax=[1,2,3], eps=1e-3):
    m = mean_with_list_axis(dataset, ax)
    shape = [1 for i in range(len(ax)+1)]
    shape[0]=dataset.shape[0]
    broadcast_m = m.reshape(shape)
    return dataset-broadcast_m

def get_next_batch(dataset, batch_size, final_size, preprocessing_func, preprocessing_args):
    # Get next batch
    batch,batch_targets = dataset.get_batch()
    # Pre-processing
    processed_batch = np.zeros((batch.shape[0],final_size[2],final_size[0],final_size[1]),
                                   dtype="float32")
    for k in range(batch_size):
        processed_batch[k] = preprocessing_func(batch[k], *preprocessing_args).transpose(2,0,1)
    # Convert labels
    labels = convert_labels(batch_targets)
    return processed_batch, labels

def images_generator(data_access, dataset, targets, batch_size, tmp_size, final_size, bagging_size, bagging_iterator,
                     multiple_input, division, preprocessing_func, preprocessing_args):
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
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets,
                                        division=division)
    elif data_access=="fuel":
        train_dataset = FuelDataset("train", tmp_size, batch_size=batch_size, bagging=bagging_size,
                                    bagging_iterator=bagging_iterator, division=division)
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)
    while 1:
        # Get next batch
        processed_batch, labels = get_next_batch(train_dataset, batch_size, final_size, preprocessing_func, preprocessing_args)
        if multiple_input == 1:
            yield processed_batch,labels
        else:
            yield [processed_batch for i in range(multiple_input)],labels

def features_generator(data_access, dataset, targets, batch_size, tmp_size, final_size, bagging_size, bagging_iterator,
                     multiple_input, preprocessing_func, preprocessing_args, pretrained_model):
    # Instantiate the dataset
    if data_access=="in-memory":
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets)
    elif data_access=="fuel":
        train_dataset = FuelDataset("train", tmp_size, batch_size=batch_size, bagging=bagging_size,
                                    bagging_iterator=bagging_iterator)
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)
    # Generator loop
    while 1:
        # Get next batch
        processed_batch, labels = get_next_batch(train_dataset, batch_size, final_size, preprocessing_func, preprocessing_args)
        if multiple_input == 1:
            features = pretrained_model.predict(processed_batch)
            yield features, labels
        else:
            features = pretrained_model.predict([processed_batch for i in range(multiple_input)])
            yield features, labels

def multi_features_generator(data_access, dataset, targets, batch_size, tmp_size, final_size, bagging_size, bagging_iterator,
                     multiple_input, preprocessing_func, preprocessing_args, pretrained_models, mode="concat"):
    # Instantiate the dataset
    if data_access=="in-memory":
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets)
    elif data_access=="fuel":
        train_dataset = FuelDataset("train", tmp_size, batch_size=batch_size, bagging=bagging_size,
                                    bagging_iterator=bagging_iterator)
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)
    # Generator loop
    while 1:
        # Get next batch
        processed_batch, labels = get_next_batch(train_dataset, batch_size, final_size, preprocessing_func, preprocessing_args)
        if multiple_input == 1:
            features = []
            for pretrained_model in pretrained_models:
                features.append(pretrained_model.predict(processed_batch, batch_size=1))
            if mode == "concat":
                features = np.concatenate(features, axis=1)
            yield features, labels
        else:
            raise Exception("Generator does not work with multiple inputs")


def check_preprocessed_data(data_access, dataset, targets, batch_size, tmp_size, final_size, preprocessing_func,
                            preprocessing_args, n=10):
    if data_access=="in-memory":
        train_dataset = InMemoryDataset("train", source=dataset, batch_size=batch_size, source_targets=targets)
    elif data_access=="fuel":
        train_dataset = FuelDataset("test", tmp_size, batch_size=batch_size, division="leaderboard", shuffle=False)
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
