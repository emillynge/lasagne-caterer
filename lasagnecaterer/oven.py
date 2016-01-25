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
from typing import List, Union
import random
from contextlib import contextmanager

# pip packages
from theano import tensor as T
import theano
import lasagne as L
import numpy as np
import requests

# github packages
from elymetaclasses.events import ChainedProps, args_from_opt
from elymetaclasses.annotations import SingleDispatch, LastResort, type_assert
from elymetaclasses.abc import io as ioabc

# relative imports
from .fridge import JsonSaveLoadMixin
from .utils import (from_start, any_to_stream)
DEBUGFLAGS = ['nocache']



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
        return cls(data, max_i=max_i)

    def to_dict(self):
        return {'data': self.data, 'max_i': self.max_i}

    @classmethod
    def from_text(cls, inp):
        charmap = cls()
        charmap.train(any_to_stream(inp))
        return charmap

    def train(self, stream: ioabc.InputStream):
        chars = set(c for c in stream.read())
        sorted_chars = sorted(chars)
        self.resize(len(sorted_chars))
        for i, c in enumerate(sorted_chars):
            self[c] = i

SplitStreams = namedtuple('SplitStreams', 'train val test')

class BatchBuffer(io.StringIO):
    class SD(SingleDispatch):

        @staticmethod
        def _input2sequence(inp: LastResort):
            logging.log('Cannot determine input type')
            return inp

        @staticmethod
        def _input2sequence(inp: Sequence):
            return inp

        @staticmethod
        def _input2sequence(inp: str):
            return inp

        @staticmethod
        def _input2sequence(inp: ioabc.SeekableInputStream):
            inp.seek(0)
            return inp.read()

        @staticmethod
        def _input2sequence(inp: ioabc.InputStream):
            return inp.read()

    def __init__(self, inp, start=None, end=None):
        start = start or 0
        end = end or -1
        super().__init__(self.SD._input2sequence(inp)[start:end])
        self.l = self.seek(0, 2)
        self.seek(0)
        self.c = Counter()
        self.mode = 'simple'
        self.batch_gen_func = None
        self.__read = self.read

    def __len__(self):
        return self.l

    class DummyProgressBar(object):
        def __init__(self, max_val=None):
            self.value = 0

        def start(self):
            pass

        def __next__(self):
            pass

        def __iter__(self):
            return self

        def __call__(self, it):
            return it

    def cast_at_read(self, castfun):
        orig_read = super().read
        def read(*args, **kwargs):
            return castfun(orig_read(*args, **kwargs))

        setattr(self, 'read', read)

    def random_seek(self, margin_left=0, margin_right=0):
        idx = random.randint(margin_left, self.l - margin_right)
        self.seek(idx)

    def splits_idx(self, sub_stream_propotions: Sequence):
        l = self.seek(0, 2)
        total_prop = sum(sub_stream_propotions)
        normed_props = [0] + [prop/total_prop for prop in sub_stream_propotions]
        fence_posts = [sum(normed_props[:i]) for i in range(1, len(normed_props) + 1)]
        return [int(l * fence_posts) for fence_posts in fence_posts]

    def split(self, sub_stream_propotions: Sequence) -> List[io.StringIO]:
        splits_idx = self.splits_idx(sub_stream_propotions)
        return [self.from_self(self.getvalue(), start=i1, end=i2) for i1, i2 in zip(splits_idx[:-1], splits_idx[1:])]

    def split3way(self, sub_stream_propotions: Sequence) -> SplitStreams:
        return SplitStreams(*self.split(sub_stream_propotions))

    @classmethod
    def from_self(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class BufferedBatchGenerator(ChainedProps):
    debug = DEBUGFLAGS

    @property
    def input_stream(self, inp):
        return any_to_stream(inp)


    @property
    def bg(self):
        return BatchBuffer(self.input_stream)

    @property
    def stream_lengths(self, splits='8,1,1'):
        splits = [int(s) for s in splits.split(',')]
        idxs = self.bg.splits_idx(splits)
        return [j-i for i, j in zip(idxs[:-1], idxs[1:])]

    @property
    def split_streams(self, splits='8,1,1'):
        splits = [int(s) for s in splits.split(',')]
        bg = self.bg
        assert isinstance(bg, BatchBuffer)
        SplitStreams = namedtuple('SplitStreams', 'train val test')
        return SplitStreams(*bg.split(splits))

    @property
    def modes(self, batch_modes='random,simple,simple'):
        BatchModes = namedtuple('BatchModes', 'train val test')
        return BatchModes(*batch_modes.strip().split(','))

    @property
    def train(self, batch_fun_train='batches_fp'):
        bg = self.split_streams.train
        mode = self.modes.train
        bg.configure(batch_fun_train, mode=mode)
        return bg

    @property
    def val(self, batch_fun_test='batches_fp'):
        bg = self.split_streams.val
        mode = self.modes.val
        bg.configure(batch_fun_test, mode=mode)
        return bg

    @property
    def test(self, batch_fun_test='batches_fp'):
        bg = self.split_streams.test
        mode = self.modes.test
        bg.configure(batch_fun_test, mode=mode)
        return bg


class CharmappedBatchGenerator(BufferedBatchGenerator):
    debug = DEBUGFLAGS

    @property
    def charmap(self, charmap=None) -> CharMap:
        if not charmap:
            with from_start(self.input_stream) as inp:
                return CharMap.from_text(inp)

        stream = any_to_stream(charmap, force_seekable=True)
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


class FullArrayBatchGenerator(CharmappedBatchGenerator):
    debug = DEBUGFLAGS
    def get_array(self, part):
        stream = getattr(self.split_streams, part)
        stream.seek(0)
        return np.array(list(stream.read()), dtype=np.bool).flatten()

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
        return self.train_full[:self.bools_per_batch + 1].flatten()

    @property
    def val_view_flat(self):
        return self.val_full[:self.bools_per_batch + 1].flatten()

    @property
    def test_view_flat(self):
        return self.test_full[:self.bools_per_batch + 1].flatten()

    @args_from_opt(1)
    def reshape2xy_batch_chunk(self, flat, batch_sz, seq_len, win_sz, features, seq_overlap=True, batch_overlap=True):
        """
        Reshape a flat array into (x,y) pair of sequences
        :param flat: flat view of current batch chunk
        :param batch_sz: number of sequences in a batch
        :param seq_len: timesteps in a sequence
        :param win_sz: Not implemented == 1
        :param features: number of classes for input and output
        :param seq_overlap: Whether sequences can overlap
        :param batch_overlap: Whether batches can overlap
        :return:
        """
        if not(seq_overlap and batch_overlap and win_sz == 1):
            raise NotImplementedError('This generator currently only works with overlap and win_sz==1')

        x = flat[:-features]    # all, excluding last element which has len == features
        y = flat[features:]  # all, excluding first element which has len == features
        return (x.reshape2xy_batch_chunk((batch_sz, seq_len, features)),
                y.reshape2xy_batch_chunk((batch_sz, seq_len, features)))

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

    @property
    def batches_per_epoch(self):
        cpb = self.chars_per_batch
        BPE = namedtuple('BPE', 'train val test')
        return BPE(*(l // cpb for l in self.stream_lengths))


    def iter(self, epochs, part='train'):
        # batch info
        bpe = getattr(self.batches_per_epoch, part)
        mode = getattr(self.modes, part)

        # batch arrays
        full = getattr(self, part + '_full')
        flat = getattr(self, part + '_view_flat')
        reshaped = getattr(self, part)

        # total length of flat
        bpb = self.bools_per_batch

        max_i = (bpe - 1) * bpb
        i = -bpb
        for e in range(epochs):
            for b in range(bpe):
                if mode == 'simple':
                    i += bpb
                elif mode == 'random':
                    i = np.random.random_integers(0, max_i)
                flat[:] = full[i:i+bpb]
                yield reshaped

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



