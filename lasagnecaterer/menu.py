"""
Easy overview of what the caterer has to offer
Hide complexity from the client. Lots of packaged deals.
"""

# builtins
from collections import (namedtuple, OrderedDict, defaultdict)
from collections.abc import Sequence
from abc import (ABC, abstractmethod, abstractclassmethod)
import json
import os
import logging
from zipfile import ZipFile, ZipInfo, ZIP_BZIP2, ZIP_STORED
import pickle
import warnings
import sys
import io
from typing import (List, Tuple)
from argparse import ArgumentParser

# pip packages
from theano.compile import SharedVariable
from numpy import ndarray

# github packages
from elymetaclasses.abc import io as ioabc
from elymetaclasses.utils import Options as _Options

# relative
from .utils import any_to_stream


class Choices(ArgumentParser):
    def __init__(self, *argparseargs, **choices):
        """
        :param choices:
                key: shortarg and dest
                value: longarg
        :return:
        """
        super().__init__(*argparseargs)
        for shortarg, longarg in choices.items():
            self.add_argument('-' + shortarg, '--' + longarg, dest=shortarg,
                              action='store_true')


from .fridge import JsonSaveLoadMixin, UniversalFridgeLoader


class Options(_Options, JsonSaveLoadMixin):
    @classmethod
    def from_dict(cls, opt_list=List[Tuple]):
        return cls.make(opt_list)

    def to_dict(self) -> dict:
        return {'opt_list': list(self.items())}

    @classmethod
    def make(cls, *args, **kwargs):
        """
        Make a new instance of Options.
        Always use this method
        :param args:
        :param kwargs:
        :return:
        """
        class Options(cls):
            pass
        kwargs['_make_called'] = True
        return Options(*args, **kwargs)


### Concrete

from .oven import FullArrayBatchGenerator
from . import recipe, fridge, oven, cook


def basic(opt):
    return fridge.BaseFridge(opt, FullArrayBatchGenerator, recipe.LSTMBase,
                             cook.CharmapCook)

def empty_fridge(file):
    return UniversalFridgeLoader.load(file)


def empty_and_swap(file, *class_swaps, **overrides):
    for klass in class_swaps:
        if issubclass(klass, recipe.LasagneBase):
            overrides['recipe_cls'] = klass
        elif issubclass(klass, oven.BufferedBatchGenerator):
            overrides['oven_cls'] = klass
        elif issubclass(klass, cook.BaseCook):
            overrides['cook_cls'] = klass
    return UniversalFridgeLoader.load(file, data_overrides=overrides)
