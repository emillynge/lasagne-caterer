"""
For orchestrating the lasagna making process
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

# pip packages
from theano.compile import SharedVariable
from numpy import ndarray


# github packages
from elymetaclasses.abc import io as ioabc

# relative
from .utils import any_to_stream
from .oven import BufferedBatchGenerator
from .recipe import LasagneBase
from .fridge import BaseFridge

class BaseCook:
    def __init__(self, oven: BufferedBatchGenerator,
                 recipe: LasagneBase,
                 fridge: BaseFridge):

        self.oven = oven
        self.recipe = recipe
        self.fridge = fridge
