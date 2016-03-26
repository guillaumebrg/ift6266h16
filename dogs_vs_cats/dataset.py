__author__ = 'Guillaume'

import numpy as np
from PIL import Image
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
#from testing import mean_with_list_axis

def dataset_division(Ntrain=17500, Nvalid=3750, Ntest=3750, seed=123456):
    np.random.seed(seed)
    index = np.arange(0,Ntrain+Nvalid+Ntest,1)
    np.random.shuffle(index)
    return index[0:Ntrain], index[Ntrain:(Ntrain+Nvalid)], index[(Ntrain+Nvalid):(Ntrain+Nvalid+Ntest)]

def dataset_division_leaderboard():
    # Returns trainset and validset indices
    indices = np.arange(25000)
    np.random.RandomState(123522).shuffle(indices)
    return  indices[0:-2500], indices[-2500:]

class Dataset(object):
    def __init__(self, source, batch_size=1, source_targets=None, shuffle=True):
        self.batch_size = batch_size
        self.current_batch = None
        self.shuffle = shuffle
        self.dataset = self.load_dataset(source)
        self.targets = self.load_targets(source_targets)

    def load_targets(self, source):
        return

    def load_dataset(self, source):
        return

    def on_new_epoch(self):
        return

    def next_batch(self):
        return

    def get_batch(self):
        self.next_batch()
        return self.current_batch

class InMemoryDataset(Dataset):
    def __init__(self, mode, source, batch_size=1, source_targets=None, shuffle=True, division="leaderboard"):
        self.mode = mode
        super( InMemoryDataset, self).__init__(source, batch_size, source_targets, shuffle)
        self.offset = 0
        self.index = np.arange(0,self.dataset.shape[0],1)
        self.iterator = 0
        self.division = division
        self.on_new_epoch()

    def load_dataset(self, source):
        dataset = np.load(source)
        if self.division == "leaderboard":
            index_train, index_valid = dataset_division_leaderboard()
            index_test = index_valid
        else:
            index_train, index_valid, index_test = dataset_division()
        if self.mode == "train":
            return dataset[index_train]
        if self.mode == "valid":
            return dataset[index_valid]
        if self.mode == "test":
            return dataset[index_test]
        else:
            raise Exception("Mode not understood. Use : train, valid or test. Here : %s"%self.mode)

    def load_targets(self, source_targets):
        if source_targets is None:
            raise Exception("source_targets not defined.")
        return self.load_dataset(source_targets)

    def on_new_epoch(self):
        if self.shuffle:
             np.random.shuffle(self.index)

    def next_batch(self):
        if (self.iterator+1)*self.batch_size+self.offset < self.dataset.shape[0]:
            s = slice(self.iterator*self.batch_size+self.offset, ((self.iterator+1)*self.batch_size)+self.offset)
            batch = self.dataset[self.index[s]]
            batch_targets = self.targets[self.index[s]]
            self.current_batch = (batch, batch_targets)
            self.iterator += 1
        else: # end epoch
            self.on_new_epoch()
            self.iterator = 0
            self.next_batch()



class FuelDataset(Dataset):
    def __init__(self, mode, tmp_size, source="image_features", batch_size=1, bagging=1, bagging_iterator=0,
                 source_targets=None, shuffle=True, division="leaderboard", N=None):
        self.mode = mode
        self.tmp_size = tmp_size
        self.bagging = bagging
        self.bagging_iterator = bagging_iterator
        self.division = division
        self.N = N
        super( FuelDataset, self).__init__(source, batch_size, source_targets, shuffle)
        self.on_new_epoch()

    def load_dataset(self, source):
        # Splitting the dataset
        if self.division == "leaderboard":
            index_train, index_valid = dataset_division_leaderboard()
        else:
            index_train, index_valid, index_test = dataset_division()
        for i in range(self.bagging_iterator):
            np.random.shuffle(index_train)
        index_train = index_train[0:int(self.bagging*index_train.shape[0])]
        if self.mode == "train":
            dataset = DogsVsCats(('train',))
            if self.shuffle:
                scheme=ShuffledScheme(index_train,self.batch_size)
            else:
                scheme=SequentialScheme(index_train,self.batch_size)
        elif self.mode == "valid":
            dataset = DogsVsCats(('train',))
            if self.shuffle:
                scheme=ShuffledScheme(index_valid,self.batch_size)
            else:
                scheme=SequentialScheme(index_valid,self.batch_size)
        elif self.mode == "test":
            if self.division == "leaderboard":
                self.shuffle = False
                dataset = DogsVsCats(('test',))
                scheme=SequentialScheme(range(12500),self.batch_size)
            else:
                dataset = DogsVsCats(('train',))
                if self.shuffle:
                    scheme=ShuffledScheme(index_test,self.batch_size)
                else:
                    scheme=SequentialScheme(index_test,self.batch_size)
        else:
            raise Exception("Mode not understood. Use : train, valid or test. Here : %s"%self.mode)
        stream = DataStream(dataset,iteration_scheme=scheme)
        if self.tmp_size[2]==1:
            downscaled_stream = ResizeAndGrayscaleImage(stream, self.tmp_size[0:2],
                                                        resample="bicubic", which_sources=(source,))
        elif self.tmp_size[2]==3:
            downscaled_stream = ResizeImage(stream, self.tmp_size[0:2],
                                            resample="bicubic", which_sources=(source,))
        else:
            raise Exception("tmp_size[2] = 1 or 3. Here : %d"%self.tmp_size[2])
        return downscaled_stream

    def load_targets(self, source_targets):
        return None

    def on_new_epoch(self):
        self.iterator = self.dataset.get_epoch_iterator()

    def next_batch(self):
        try:
            batch,targets = next(self.iterator)
        except StopIteration:
            self.iterator = self.dataset.get_epoch_iterator()
            batch,targets = next(self.iterator)
        # Reshape
        batch = np.concatenate([b.transpose(1,2,0)[None,:,:] for b in batch])
        if batch.shape[0]!=self.batch_size: # to always get batch of same size
            self.next_batch()
        else:
            self.current_batch = (batch, targets)

    def return_whole_dataset(self):
        self.on_new_epoch()
        for i, batch in enumerate(self.iterator):
            if i == 0:
                dataset = np.concatenate([b.transpose(1,2,0)[None,:,:] for b in batch[0]])
                targets = batch[1]
            else:
                dataset = np.concatenate((dataset,np.concatenate([b.transpose(1,2,0)[None,:,:] for b in batch[0]])))
                targets = np.concatenate((targets,batch[1]))
        return dataset, targets

def extrapolate(im):
    """Duplicate the last row/col to make im be squared."""
    rows, cols = im.shape[0:2]
    if rows > cols:
        out = np.concatenate((np.repeat(im[:,0:1], (rows-cols)/2, axis=1), im), axis=1)
        out = np.concatenate((out, np.repeat(im[:,-1:], rows-out.shape[1], axis=1)), axis=1)
    else:
        out = np.concatenate((np.repeat(im[0:1,:], (cols-rows)/2, axis=0), im), axis=0)
        out = np.concatenate((out, np.repeat(im[-1:,:], cols-out.shape[0], axis=0)), axis=0)
    return out

def zero_padding(im):
    """0-padding to make im be squared."""
    rows, cols = im.shape[0:2]
    if rows > cols:
        out = np.concatenate((np.repeat(im[:,0:1]*0, (rows-cols)/2, axis=1), im), axis=1)
        out = np.concatenate((out, np.repeat(im[:,-1:]*0, rows-out.shape[1], axis=1)), axis=1)
    else:
        out = np.concatenate((np.repeat(im[0:1,:]*0, (cols-rows)/2, axis=0), im), axis=0)
        out = np.concatenate((out, np.repeat(im[-1:,:]*0, cols-out.shape[0], axis=0)), axis=0)
    return out

# def random_padding(im):
#     """random-padding the last row/col to make im be squared."""
#     rows, cols = im.shape[0:2]
#     m = mean_with_list_axis(im, [0,1])
#     std = mean_with_list_axis(np.square(im-m), [0,1])
#     std = np.sqrt(std)
#     if rows > cols:
#         out = np.concatenate((np.abs(np.random.randn(rows, (rows-cols)/2, im.shape[2])*std+m), im), axis=1)
#         out = np.concatenate((out, np.abs(np.random.randn(rows, rows-out.shape[1], im.shape[2])*std+m)), axis=1)
#     else:
#         out = np.concatenate((np.abs(np.random.randn((cols-rows)/2, cols, im.shape[2])*std+m), im), axis=0)
#         out = np.concatenate((out, np.abs(np.random.randn(cols-out.shape[0], cols, im.shape[2])*std+m)), axis=0)
#     return np.array(out, "uint8")

class ResizeImage(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to minimum dimensions.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    new_shape : 2-tuple
        The shape `(height, width)` dimensions every image must have..
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.

    """
    def __init__(self, data_stream, new_shape, resample='bicubic',
                 **kwargs):
        self.new_shape = new_shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ResizeImage, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        dt = example.dtype
        width, height = self.new_shape
        if example.ndim == 3:
            im = example.transpose(1, 2, 0)
        else:
            im = example
        #im = random_padding(im)
        im = Image.fromarray(im)
        im = np.array(im.resize((width, height))).astype(dt)
        # If necessary, undo the axis swap from earlier.
        if im.ndim == 3:
            example = im.transpose(2, 0, 1)
        else:
            example = im
        return example

class ResizeAndGrayscaleImage(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to minimum dimensions.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    new_shape : 2-tuple
        The shape `(height, width)` dimensions every image must have..
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.

    """
    def __init__(self, data_stream, new_shape, resample='nearest',
                 **kwargs):
        self.new_shape = new_shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ResizeAndGrayscaleImage, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        dt = example.dtype
        width, height = self.new_shape
        if example.ndim == 3:
            im = example.transpose(1, 2, 0)
        else:
            im = example
        #im = extrapolate(im)
        im = Image.fromarray(im).convert("L")
        im = np.array(im.resize((width, height))).astype(dt)
        example = im[None,:,:]
        return example



