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

# pip packages
from theano.compile import SharedVariable
from numpy import ndarray


# github packages
from elymetaclasses.abc import io as ioabc
from elymetaclasses.utils import Options as _Options

# relative
from .utils import any_to_stream
from .fridge import JsonSaveLoadMixin


class Options(_Options, JsonSaveLoadMixin):
    @classmethod
    def from_dict(cls, opt_list=List[Tuple]):
        return cls.make(opt_list)

    def to_dict(self) -> dict:
        return {'opt_list': list(self.items())}