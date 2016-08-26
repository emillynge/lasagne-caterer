"""
For preparing lasagnas (training)
"""

# builtins
import json
from collections import (namedtuple, defaultdict, Counter, UserDict, OrderedDict)
from collections.abc import MutableSequence
from functools import (lru_cache, partial, singledispatch)
import os
import io
import _pyio
import logging
from collections.abc import (Sequence)
from itertools import chain
from typing import List, Union
import random
from contextlib import contextmanager

# pip packages
import struct
from theano import tensor as T
import theano
import lasagne as L
import numpy as np
import requests
from threading import Lock
from _thread import LockType

# github packages
from elymetaclasses.events import ChainedProps, args_from_opt
from elymetaclasses.annotations import SingleDispatch, LastResort, type_assert
from elymetaclasses.abc import io as ioabc

# relative imports
from .utils import (from_start, any_to_char_stream)
from .fridge import JsonSaveLoadMixin, ClassSaveLoadMixin

DEBUGFLAGS = []#'nocache']


class CharMap(OrderedDict, JsonSaveLoadMixin):
    def __init__(self, *args, max_i=None, **kwargs):
        self.reverse = dict()
        super().__init__(*args, **kwargs)
        self.max_i = max_i or len(self)
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def resize(self, max_i):
        self.max_i = max_i
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.reverse[value] = key

    def __missing__(self, char):
        i = len(self)
        if self.max_i <= i:
            raise ValueError('No more room')
        self[char] = i
        return i

    def str2arraygen(self, s):
        for char in s:
            yield self.vectors[self[char]]

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) > 1:
                return ''.join(self(vec) for vec in inp)
            if inp.dtype == np.bool:
                return self.reverse[int(self.idx_vec[inp])]
            return ''.join(self.reverse[int(i)] for i in inp)

        if isinstance(inp, MutableSequence):
            return np.concatenate(tuple(self(c) for c in inp)).reshape((len(inp), self.max_i))

        if isinstance(inp, str):
            return self.vectors[self[inp]]

        if isinstance(inp, (np.int, np.int64)):
            return self.reverse[int(inp)]

    def truncate(self):
        self.resize(len(self))

    @classmethod
    def from_dict(cls, data: dict, max_i=None):
        return cls(max_i=max_i, **data)

    def to_dict(self):
        return {'data': dict(self), 'max_i': self.max_i}

    @classmethod
    def from_text(cls, inp):
        charmap = cls()
        charmap.train(any_to_char_stream(inp))
        return charmap

    def train(self, stream: ioabc.InputStream):
        chars = set(c for c in stream.read())
        sorted_chars = sorted(chars)
        self.resize(len(sorted_chars))
        for i, c in enumerate(sorted_chars):
            self[c] = i

SplitStreams = namedtuple('SplitStreams', 'train val test')


class InputBuffer:
    class SD(SingleDispatch):

        @classmethod
        def _input2sequence(cls, inp: LastResort):
            logging.log('Cannot determine input type')
            return inp

        @classmethod
        def _input2sequence(cls, inp: Sequence):
            return inp

        @classmethod
        def _input2sequence(cls, inp: str):
            return inp

        @classmethod
        def _input2sequence(cls, inp: ioabc.SeekableInputStream):
            inp.seek(0)
            return inp.read()

        @classmethod
        def _input2sequence(cls, inp: ioabc.InputStream):
            return inp.read()

    def __init__(self, inp, start=None, end=None):
        super().__init__()
        start = start or 0
        end = end or -1
        self.write(self.SD._input2sequence(inp)[start:end])
        self.seek(0)
        self.c = Counter()
        self.mode = 'simple'
        self.batch_gen_func = None
        self.__read = self.read

    def __len__(self):
        with self.from_start():
            return self.seek(0, 2)

    def cast_at_read(self, castfun):
        orig_read = super().read
        def read(*args, **kwargs):
            return castfun(orig_read(*args, **kwargs))

        setattr(self, 'read', read)

    def random_seek(self, margin_left=0, margin_right=0):
        idx = random.randint(margin_left, self.l - margin_right)
        self.seek(idx)

    def splits_idx(self, sub_stream_propotions: Sequence):
        l = len(self)
        total_prop = sum(sub_stream_propotions)
        normed_props = [0] + [prop/total_prop for prop in sub_stream_propotions]
        fence_posts = [sum(normed_props[:i]) for i in range(1, len(normed_props) + 1)]
        return [int(l * fence_posts) for fence_posts in fence_posts]

    @contextmanager
    def from_start(self):
        pos = self.tell()
        self.seek(0)
        yield
        self.seek(pos)

    def split(self, sub_stream_propotions: Sequence) -> List[io.StringIO]:
        splits_idx = self.splits_idx(sub_stream_propotions)
        read_len = [s2 - s1 for s1, s2 in zip(splits_idx[:-1], splits_idx[1:])]
        with self.from_start():
            return [self.from_self(self.__read(n)) for n in read_len]

    def split3way(self, sub_stream_propotions: Sequence) -> SplitStreams:
        return SplitStreams(*self.split(sub_stream_propotions))

    @classmethod
    def from_self(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class StringBuffer(InputBuffer, io.StringIO):
    pass


class _FloatBuffer(io.BytesIO):
    item_sz = 4

    def write(self, float_seq):
        if isinstance(float_seq, bytes):
            return super().write(float_seq)
        n_fl = len(float_seq)
        b = struct.pack('f' * n_fl, *float_seq)
        super().write(b)
        return n_fl

    def read(self, n=-1):
        b = super().read(self.item_sz * n)

        return memoryview(b).cast('f').tolist()

    def tell(self, *args, **kwargs):
        return super().tell(*args, **kwargs) // self.item_sz

    def seek(self, pos, whence=0, **kwargs):
        pos *= self.item_sz
        pos = super().seek(pos, whence)
        return pos // self.item_sz


class FloatBuffer(InputBuffer, _FloatBuffer):
    def __init__(self, inp, start=None, end=None):
        start = start if start is None else start * self.item_sz
        end = end if end is None else end * self.item_sz
        super().__init__(inp, start=start, end=end)

    class SD(InputBuffer.SD):

        @staticmethod
        def file2seq(file):
            import csv
            if file.name.endswith('.npy'):
                return np.load(file).flatten().tolist()

            if file.name.endswith('.csv'):
                f = io.TextIOWrapper(file)
                return list(float(i) for i in chain(*(row for row in csv.reader(f))))

        @classmethod
        def _input2sequence(cls, inp: LastResort):
            return inp

        @classmethod
        def _input2sequence(cls, inp: ioabc.SeekableInputStream):
            inp.seek(0)
            return cls.file2seq(inp)

        @classmethod
        def _input2sequence(cls, inp: ioabc.InputStream):
            return cls.file2seq(inp)


class _FloatArrayBuffer(io.BytesIO):
    item_sz = 4

    def __init__(self, *args, **kwargs):
        self.array_len = None
        self._bytes_per_array = None
        super().__init__(*args, **kwargs)

    @property
    def bytes_per_array(self):
        return self.item_sz * self.array_len

    def write(self, array_seq):
        if isinstance(array_seq, bytes):
            return super().write(array_seq)
        n_arr = len(array_seq)
        if not self.array_len:
            try:
                self.array_len = len(array_seq[0])
            except TypeError:
                self.array_len = 1

        inp = chain(*array_seq) if isinstance(array_seq[0],
                                              Sequence) else array_seq
        b = struct.pack('f' * (n_arr * self.array_len), *inp)
        super().write(b)

    def read(self, n=-1):
        b = super().read(self.bytes_per_array * n)
        n_array = len(b) // self.bytes_per_array
        return memoryview(b).cast('f', [n_array, self.array_len]).tolist()

    def tell(self, *args, **kwargs):
        return super().tell(*args, **kwargs) // self.bytes_per_array

    def seek(self, pos, whence=0, **kwargs):
        pos *= self.bytes_per_array
        pos = super().seek(pos, whence)
        return pos // self.bytes_per_array


class FloatArrayBuffer(InputBuffer, _FloatArrayBuffer):
    def __init__(self, inp, start=None, end=None):
        start = start if start is None else start * self.item_sz
        end = end if end is None else end * self.item_sz
        super().__init__(inp, start=start, end=end)

    class SD(InputBuffer.SD):

        @staticmethod
        def file2seq(file):
            import csv
            if file.name.endswith('.npy'):
                npfile = np.load(file)
                try:
                    return npfile.flatten().tolist()
                except AttributeError:
                    return next(iter(npfile.items()))[1].flatten().tolist()

            if file.name.endswith('.csv'):
                f = io.TextIOWrapper(file)
                return list(list(float(i) for i in row) for row in csv.reader(f))

        @classmethod
        def _input2sequence(cls, inp: LastResort):
            return inp

        @classmethod
        def _input2sequence(cls, inp: ioabc.SeekableInputStream):
            inp.seek(0)
            return cls.file2seq(inp)

        @classmethod
        def _input2sequence(cls, inp: ioabc.InputStream):
            return cls.file2seq(inp)

from .utils import ChainPropsABCMetaclass


class BufferedBatchGenerator(ChainedProps, ClassSaveLoadMixin, metaclass=ChainPropsABCMetaclass):
    debug = DEBUGFLAGS

    @property
    def input_stream(self, inp):
        return inp

    @property
    def bg(self):
        return InputBuffer(self.input_stream)

    @property
    def stream_lengths(self, splits='8,1,1'):
        splits = [int(s) for s in splits.split(',')]
        idxs = self.bg.splits_idx(splits)
        return [j-i for i, j in zip(idxs[:-1], idxs[1:])]

    @property
    def split_streams(self, splits='8,1,1'):
        splits = [int(s) for s in splits.split(',')]
        bg = self.bg
        assert isinstance(bg, InputBuffer)
        SplitStreams = namedtuple('SplitStreams', 'train val test')
        return SplitStreams(*bg.split(splits))

    @property
    def modes(self, batch_modes='random,simple,simple'):
        BatchModes = namedtuple('BatchModes', 'train val test')
        return BatchModes(*batch_modes.strip().split(','))


class CharBatchMixin(metaclass=ChainPropsABCMetaclass):
    @property
    def input_stream(self, inp):
        return any_to_char_stream(inp)

    @property
    def bg(self):
        return StringBuffer(self.input_stream)


class FloatBatchMixin(metaclass=ChainPropsABCMetaclass):
    @property
    def input_stream(self, inp):
        return any_to_char_stream(inp, bytes)

    @property
    def bg(self):
        return FloatBuffer(self.input_stream)


class FloatArrayBatchMixin(FloatBatchMixin):
    @property
    def bg(self, features):
        bg = FloatArrayBuffer(self.input_stream)
        bg.array_len = features
        return bg



class CharmappedGeneratorMixin(CharBatchMixin):
    debug = DEBUGFLAGS

    @property
    def charmap(self, charmap=None) -> CharMap:
        if not charmap:
            with from_start(self.input_stream) as inp:
                return CharMap.from_text(inp)

        stream = any_to_char_stream(charmap, force_seekable=True)
        try:
            return CharMap.load(stream)
        except json.JSONDecodeError as e:
            raise ValueError('Could not read {} as a charmap'.format(charmap)) from e

    @property
    def split_streams(self, splits='8,1,1'):
        splits = [int(s) for s in splits.split(',')]
        cm = self.charmap
        streams = self.bg.split3way(splits)
        for stream in streams:
            stream.cast_at_read(cm.str2arraygen)
        return streams


class FullArrayBatchGenerator(BufferedBatchGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locks = dict((part, Lock()) for part in ['train', 'val', 'test'])

    def get_lock(self, part) -> LockType:
        """

        :rtype: LockType
        """
        return self.locks[part]

    debug = DEBUGFLAGS
    @args_from_opt(1)
    def get_array(self, part, dtype='bool'):
        stream = getattr(self.split_streams, part)
        stream.seek(0)
        return np.array(list(stream.read()), dtype=getattr(np, dtype)).flatten()

    @property
    def train_full(self):
        return self.get_array('train')

    @property
    def val_full(self):
        return self.get_array('val')

    @property
    def test_full(self):
        return self.get_array('test')

    @property
    def train_view_flat(self):
        return self.train_full[:self.bools_per_batch].flatten()

    @property
    def val_view_flat(self):
        return self.val_full[:self.bools_per_batch].flatten()

    @property
    def test_view_flat(self):
        return self.test_full[:self.bools_per_batch].flatten()

    @args_from_opt(1)
    def reshape2xy_batch_chunk(self, flat, batch_sz, seq_len, win_sz, features,
                               seq_overlap=True, batch_overlap=True, complete=False):
        """
        Reshape a flat array into (x,y) pair of sequences
        :param flat: flat view of current batch chunk
        :param batch_sz: number of sequences in a batch
        :param seq_len: timesteps in a sequence
        :param win_sz: Not implemented == 1
        :param features: number of classes for input and output
        :param seq_overlap: Whether sequences can overlap
        :param batch_overlap: Whether batches can overlap
        :param complete: is this the full array? if so ignore batch_sz and seq_len
        :return:
        """

        if not(seq_overlap and batch_overlap and win_sz == 1):
            raise NotImplementedError('This generator currently only works with overlap and win_sz==1')

        x = flat[:-features]    # all, excluding last element which has len == features
        y = flat[features:]  # all, excluding first element which has len == features

        if complete:
            batch_sz = 1
            seq_len = len(y)

        return (x.reshape((batch_sz, seq_len, win_sz, features)),
                y.reshape((batch_sz, seq_len, features)))

    # noinspection PyMethodOverriding
    @property
    def train(self):
        return self.reshape2xy_batch_chunk(self.train_view_flat)

    # noinspection PyMethodOverriding
    @property
    def val(self):
        return self.reshape2xy_batch_chunk(self.val_view_flat)

    # noinspection PyMethodOverriding
    @property
    def test(self):
        return self.reshape2xy_batch_chunk(self.test_view_flat)

    # noinspection PyMethodOverriding
    @property
    def train_complete(self):
        return self.reshape2xy_batch_chunk(self.train_full.flatten(), complete=True)

    # noinspection PyMethodOverriding
    @property
    def val_complete(self):
        return self.reshape2xy_batch_chunk(self.val_full.flatten(), complete=True)

    # noinspection PyMethodOverriding
    @property
    def test_complete(self):
        return self.reshape2xy_batch_chunk(self.test_full.flatten(), complete=True)

    @property
    def batches_per_epoch(self):
        cpb = self.chars_per_batch
        BPE = namedtuple('BPE', 'train val test')
        return BPE(*(l // cpb for l in self.stream_lengths))

    @args_from_opt(1)
    def prepare_flat_array(self, part, features):
        lock = self.get_lock(part)
        if lock.locked():
            raise PermissionError('{} is already has a generator working on it.'
                                  'If you need multiple generators you should '
                                  'copy this oven'.format(part))

        with lock:
            mode = getattr(self.modes, part)
            full = getattr(self, part + '_full')
            flat = getattr(self, part + '_view_flat')

            # total length of flat
            bpb = self.bools_per_batch

            bpe = getattr(self.batches_per_epoch, part)

            max_c = ((bpe - 1) * bpb) // features

            if mode == 'simple':
                shuffle = lambda j:  j + bpb
            elif mode == 'random':
                shuffle = lambda j: np.random.random_integers(0, max_c) * features
            else:
                raise ValueError('unknown mode "{}"'.format(mode))

            try:
                i = -bpb
                while True:
                    i = shuffle(i)
                    flat[:] = full[i:i+bpb]
                    _i = yield i
                    i = _i or i  # possible to change i
            except IndexError:
                raise StopIteration('Out of bounds') from IndexError

    def iter_batch(self, batches=-1, part='train'):
        """
        Iterate over batches
        :param batches: number of batches to output. -1 means a full epoch
        :param part: train, val or test
        :return:
        """
        # batch arrays
        reshaped = getattr(self, part)
        bpe = getattr(self.batches_per_epoch, part)
        if batches < 0:
            batches = bpe

        gen = self.prepare_flat_array(part)
        for b in range(batches):
            next(gen)
            yield reshaped

    def iter_epoch(self, epochs, part='train'):
        for e in range(epochs):
            yield from self.iter_batch(part=part)

    @property
    def chars_per_batch(self, win_sz, batch_sz, seq_len, seq_overlap=True, batch_overlap=True):
        char_per_seq = seq_len if seq_overlap else seq_len + win_sz
        #char_per_seq + 1
        char_per_batch = char_per_seq * batch_sz
        char_per_batch += win_sz
        return char_per_batch

    @property
    def bools_per_batch(self, features):
        return self.chars_per_batch * features


class ContigousAcrossBatch(FullArrayBatchGenerator):
    """
    Produces batches where sequences are contiguous across batches
     input element x[b, t, s] is element in sequence #s in batch #b at time #t
     T = sequence length
     B = batch size (number of sequences in batch)

     input element in raw input stream I => I[n]

     x[b, T, s] = I[n]
     x[b + 1, 0, s] = I[n+1]

                   position       offset
     n(b, t, s) = (s * T + t) + (b * B * T)

     This means that sequences are coupled from batch b to b + 1
        y[b,T,s] == x[b+1,0,s]
     if
        x[b,t,s] predicts x[b,t + 1,s]

    """

    debug = DEBUGFLAGS
    @args_from_opt(1)
    def get_array(self, part, batch_sz, features, dtype='bool'):
        """
        Fetch stream, put into array, shuffle and save it
        :param part:
        :param batch_sz:
        :param features:
        :return:
        """
        stream = getattr(self.split_streams, part)
        stream.seek(0)
        I = np.array(list(stream.read()), dtype=getattr(np, dtype)).flatten()
        I_crop = I[:getattr(self.split_len, part) * features]
        return I_crop.reshape((batch_sz, -1, features)).transpose(1, 0, 2).flatten()

    @property
    def split_len(self, batch_sz):
        """
        Actual character length of each split (part, val train)
        :param batch_sz:
        :return:
        """
        SplitLen = namedtuple('SplitLen', 'train val test')
        l2crop = lambda l: l - (l % batch_sz)
        return SplitLen(*(l2crop(l) for l in self.stream_lengths))


    @property
    def chars_per_batch(self, batch_sz, seq_len):
        """
        Number of characters needed to form one batch
        Every sequence in batch needs 1 extra character to provide the offset
        for predictions
        :param batch_sz:
        :param seq_len:
        :param seq_overlap:
        :param batch_overlap:
        :return:
        """
        char_per_seq = seq_len + 1
        char_per_batch = char_per_seq * batch_sz
        return char_per_batch


    @property
    def batches_per_epoch(self, batch_sz):
        cpb = self.chars_per_batch

        # reduce cpb: we reuse 1 character per sequence per batch
        cpb = cpb - batch_sz
        BPE = namedtuple('BPE', 'train val test')

        def bpe_from_split_len(l):
            # fenceposting: cannot reuse last prediction of last batch
            l = l - batch_sz
            return l // cpb

        return BPE(*(bpe_from_split_len(l) for l in self.split_len))

    @args_from_opt(1)
    def reshape2xy_batch_chunk(self, flat, batch_sz, seq_len, win_sz, features,
                               seq_overlap=True, batch_overlap=True, complete=False):
        """
        Reshape a flat array into (x,y) pair of sequences
        :param flat: flat view of current batch chunk
        :param batch_sz: number of sequences in a batch
        :param seq_len: timesteps in a sequence
        :param win_sz: Not implemented == 1
        :param features: number of classes for input and output
        :param seq_overlap: Whether sequences can overlap
        :param batch_overlap: Whether batches can overlap
        :param complete: is this the full array? if so ignore batch_sz and seq_len
        :return:
        """


        if not(seq_overlap and batch_overlap and win_sz == 1):
            raise NotImplementedError('This generator currently only works with overlap and win_sz==1')

        # get correct number of elements for matrix
        crop = - (len(flat) % (features * batch_sz))

        x = flat[:-features * batch_sz + crop] # all, excluding last element which has len == features
        crop = crop or None
        y = flat[features * batch_sz:crop]  # all, excluding first element which has len == features

        if complete:
            batch_sz = 1
            seq_len = len(y)


        # do reshape and then transpose (dimshuffle) to get
        # (batch_sz, seq_len, features)
        x = x.reshape((-1, batch_sz, 1, features)).transpose(1, 0, 2, 3)
        y = y.reshape((-1, batch_sz, features)).transpose(1, 0, 2)

        return x, y

    @args_from_opt(1)
    def prepare_flat_array(self, part, features, batch_sz, seq_len):
        lock = self.get_lock(part)
        if lock.locked():
            raise PermissionError('{} is already has a generator working on it.'
                                  'If you need multiple generators you should '
                                  'copy this oven'.format(part))

        with lock:
            mode = getattr(self.modes, part)
            full = getattr(self, part + '_full')
            flat = getattr(self, part + '_view_flat')


            bpb = self.bools_per_batch

            # get bools used per batch
            bpb_u = bpb - features * batch_sz

            bpe = getattr(self.batches_per_epoch, part)



            if mode == 'simple':
                # advance seq_len minus overlap of 1 char
                shuffle = lambda j:  j + bpb_u

            elif mode == 'random':
                # jump to completely random char

                # last char idx that will leave space for a full batch
                max_c = ((bpe - 2) * bpb_u + bpb) // features
                shuffle = lambda j: np.random.random_integers(0, max_c) * features

            elif mode == 'random_start':
                # like simple, but starting position can vary up to seq_len

                # how many chars left unused if we start at 0?
                lee_way = (getattr(self.split_len, part) // batch_sz - 1) % seq_len

                def shuffle(j):
                    if j < 0:
                        return np.random.random_integers(0, lee_way) * features
                    return j + bpb_u
            else:
                raise ValueError('unknown mode "{}"'.format(mode))

            try:
                i = -bpb_u
                while True:
                    i = shuffle(i)
                    flat[:] = full[i:i+bpb]
                    _i = yield i
                    i = _i or i  # possible to change i
            except (IndexError, ValueError):
                raise StopIteration('Out of bounds') from IndexError


class Synthetics:
    class FullArrayCharbatches(CharmappedGeneratorMixin, FullArrayBatchGenerator): pass

    class ContiguousCharBatches(CharmappedGeneratorMixin, ContigousAcrossBatch): pass