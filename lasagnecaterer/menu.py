"""
Easy overview of what the caterer has to offer
Hide complexity from the client. Lots of packaged deals.
"""

# builtins
import sys
from typing import (List, Tuple)
from argparse import ArgumentParser

# pip packages

# github packages
from elymetaclasses.utils import Options as _Options

# relative

from . import recipe, fridge, oven, cook
from .oven import FullArrayBatchGenerator
from .fridge import JsonSaveLoadMixin, UniversalFridgeLoader


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

def basic(opt):
    return fridge.BaseFridge(opt, FullArrayBatchGenerator, recipe.LSTMBase,
                             cook.CharmapCook)

def empty_fridge(file, *class_swaps, **overrides):
    for klass in class_swaps:
        if issubclass(klass, recipe.LasagneBase):
            overrides['recipe_cls'] = klass
        elif issubclass(klass, oven.BufferedBatchGenerator):
            overrides['oven_cls'] = klass
        elif issubclass(klass, cook.BaseCook):
            overrides['cook_cls'] = klass
    return UniversalFridgeLoader.load(file, data_overrides=overrides)