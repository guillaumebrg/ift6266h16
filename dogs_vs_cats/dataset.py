__author__ = 'Guillaume'

import numpy as np
from PIL import Image
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer

def dataset_division(Ntrain=17500, Nvalid=3750, Ntest=3750, seed=123456):
    np.random.seed(seed)
    index = np.arange(0,Ntrain+Nvalid+Ntest,1)
    np.random.shuffle(index)
    return index[0:Ntrain], index[Ntrain:(Ntrain+Nvalid)], index[(Ntrain+Nvalid):(Ntrain+Nvalid+Ntest)]

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
    def __init__(self, mode, source, batch_size=1, source_targets=None, shuffle=True):
        self.mode = mode
        super( InMemoryDataset, self).__init__(source, batch_size, source_targets, shuffle)
        self.offset = 0
        self.index = np.arange(0,self.dataset.shape[0],1)
        self.iterator = 0
        self.on_new_epoch()

    def load_dataset(self, source):
        dataset = np.load(source)
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
    def __init__(self, mode, tmp_size, source="image_features", batch_size=1, source_targets=None, shuffle=True):
        self.mode = mode
        self.tmp_size = tmp_size
        super( FuelDataset, self).__init__(source, batch_size, source_targets, shuffle)
        self.on_new_epoch()

    def load_dataset(self, source):
        index_train, index_valid, index_test = dataset_division()
        train = DogsVsCats(('train',))
        if self.mode == "train":
            if self.shuffle:
                scheme=ShuffledScheme(index_train,self.batch_size)
            else:
                scheme=SequentialScheme(index_train,self.batch_size)
        elif self.mode == "valid":
            if self.shuffle:
                scheme=ShuffledScheme(index_valid,self.batch_size)
            else:
                scheme=SequentialScheme(index_valid,self.batch_size)
        elif self.mode == "test":
            if self.shuffle:
                scheme=ShuffledScheme(index_test,self.batch_size)
            else:
                scheme=SequentialScheme(index_test,self.batch_size)
        else:
            raise Exception("Mode not understood. Use : train, valid or test. Here : %s"%self.mode)
        stream = DataStream(train,iteration_scheme=scheme)
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
    def __init__(self, data_stream, new_shape, resample='nearest',
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
        im = Image.fromarray(im).convert("L")
        im = np.array(im.resize((width, height))).astype(dt)
        example = im[None,:,:]
        return example



