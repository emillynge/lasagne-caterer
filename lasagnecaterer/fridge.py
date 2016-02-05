"""
For storing lasagnas!
"""
# builtins
from collections import (namedtuple, OrderedDict, defaultdict)
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from abc import (ABC, abstractmethod, abstractclassmethod)
import json
import os
import logging
from pprint import pprint
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

class JsonSaveLoadMixin(ABC):
    """
    Mixin for adding save/load utility to a class.
    implementation must provide the abstract methods:
        from_dict
        to_dict

    all data in result from to_dict must be json-serializeable
    """
    def save(self, file):
        if isinstance(file, ioabc.OutputStream):
            file.write(self.dumps())
        elif isinstance(file, str):
            with open(file, 'w') as fp:
                fp.write(self.dumps())

    @classmethod
    def load(cls, file):
        if isinstance(file, ioabc.InputStream):
            return cls.from_dict(**json.load(file))
        elif isinstance(file, str):
            if os.path.isfile(file):
                with open(file, 'r') as fp:
                    return cls.from_dict(**json.load(fp))
            else:
                logging.warning('Input is a str but not a filename! trying to use loads...')
                return cls.loads(file)

    @classmethod
    def loads(cls, s):
        return cls.from_dict(**json.loads(s))

    def dumps(self):
        return json.dumps(self.to_dict())

    @abstractclassmethod
    def from_dict(cls, *args, **kwargs):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


ClassDescription = namedtuple('ClassDescription', 'modulename clsname')


class ClassDescriptionMixin:
    @classmethod
    def class_descr(cls):
        return cls.dscr_from_class(cls)

    @staticmethod
    def dscr_from_class(cls: type):
        module = sys.modules[cls.__module__]
        modulename = module.__name__
        clsname = cls.__name__
        return ClassDescription(modulename, clsname)

    @staticmethod
    def class_from_dscr(descr: ClassDescription):
        module = import_module(descr.modulename)
        return getattr(module, descr.clsname)

    @classmethod
    def class_from_fname(cls, fname, splitter):
        obj_name, ending = fname.split(splitter)
        modulename = '.'.join(ending.split('.')[:-1])
        clsname = ending.split('.')[-1]
        klass = cls.class_from_dscr(ClassDescription(modulename, clsname))
        return klass, obj_name


class ClassSaveLoadMixin(ClassDescriptionMixin, JsonSaveLoadMixin):
    def to_dict(self) -> dict:
        descr = self.dscr_from_class(self.__class__)
        return descr._asdict()

    @classmethod
    def from_dict(cls, *args, **kwargs):
        descr = ClassDescription(**kwargs)
        return cls.class_from_dscr(descr)


class ZipSaveError(Exception):
    pass

class ZipLoadError(Exception):
    pass


class SaveLoadZipFilemixin(ClassDescriptionMixin, ABC):
    pickle_classes = [SharedVariable,
                      ndarray]

    @property
    def _pickle_classes(self):
        return tuple(self.pickle_classes)


    @classmethod
    def write_main(cls):
        modulename, clsname = cls.dscr_from_class(cls)
        __main__ = ["#!/usr/bin/python3"]

        # Make sure cwd is in path
        __main__.append('import sys\nimport os')
        __main__.append("sys.path.insert(0, os.path.abspath('.'))")

        # import the correct class and use classmethod load
        __main__.append('from {0} import {1}'.format(modulename, clsname))
        __main__.append('obj = {0}.load(sys.argv[0])'.format(clsname))

        # boot up!
        __main__.append('obj.bootstrap()')
        return '\n'.join(__main__)

    @abstractmethod
    def bootstrap(self):
        pass

    def _save(self, stream, data, compression=ZIP_STORED,
              handle_unknown=warnings.warn):
        with ZipFile(stream, mode='w', compression=compression) as zf:
            for obj_name, obj in data.items():
                ext = None
                buffer = io.BytesIO()
                if isinstance(obj, ClassSaveLoadMixin):
                    ext = 'slcls'
                    strbuffer = io.TextIOWrapper(buffer)
                    obj.save(strbuffer)
                    strbuffer.flush()

                elif isinstance(obj, JsonSaveLoadMixin):
                    modulename, clsname = self.dscr_from_class(obj.__class__)
                    ext = 'sljson.' + modulename + '.' + clsname
                    strbuffer = io.TextIOWrapper(buffer)
                    obj.save(strbuffer)
                    strbuffer.flush()

                elif isinstance(obj, SaveLoadZipFilemixin):
                    modulename, clsname = self.dscr_from_class(obj.__class__)
                    ext = 'slzip.' + modulename + '.' + clsname
                    obj.save(buffer)

                elif isinstance(obj, self._pickle_classes):
                    ext = 'pkl'
                    pickle.dump(obj, buffer)

                elif isinstance(obj, Sequence) and isinstance(obj[0], self._pickle_classes):
                    ext = 'lstzip'
                    self._save(buffer,
                               dict((str(i), o) for i, o in enumerate(obj)),
                               compression=ZIP_BZIP2,
                               handle_unknown=warnings.warn)

                else:
                    strbuffer = io.TextIOWrapper(buffer)
                    try:
                        json.dump(obj, strbuffer)
                        strbuffer.flush()
                        #buffer.write(json.dumps(obj).encode())

                        ext = 'json'
                    except TypeError as e:
                        warnings.warn('"{0}" failed to be dumped to json\n\t{1}'.format(obj_name, e.args))
                        try:
                            pickle.dump(obj, buffer)
                            ext = 'pkl'
                        except pickle.PickleError:
                            warnings.warn('"{0}" failed to be dumped to pkl\n\t{1}'.format(obj_name, e.args))

                if ext is None:
                    msg = 'I tried, but alas {0} could not be stored'.format(obj_name)
                    if issubclass(handle_unknown, Exception):
                        raise handle_unknown(msg)
                    else:
                        handle_unknown(msg)

                zf.writestr(obj_name + '.' + ext, buffer.getvalue())
            zf.writestr('__main__.py', self.write_main())

    @classmethod
    def _load(cls, stream, handle_unknown=warnings.warn):
        data = OrderedDict()
        with ZipFile(stream, mode='r') as zf:
            for file_info in zf.filelist:
                fname = file_info.filename
                assert isinstance(file_info, ZipInfo)
                obj = None
                obj_name = None
                with zf.open(file_info, mode='r') as obj_stream:
                    if fname.endswith('.pkl'):
                        obj_name = fname[:-4]
                        obj = pickle.load(obj_stream)

                    elif file_info.filename.endswith('.json'):
                        obj_name = fname[:-5]
                        obj = json.loads(obj_stream.read().decode('utf8'))

                    elif file_info.filename == '__main__.py':
                        continue

                    elif '.slcls' in fname:
                        strobj_str = io.TextIOWrapper(obj_stream)
                        obj = ClassSaveLoadMixin.load(strobj_str)
                        obj_name = fname[:-6]

                    elif '.sljson.' in fname:
                        klass, obj_name = cls.class_from_fname(fname, '.sljson.')
                        strobj_str = io.TextIOWrapper(obj_stream)
                        obj = klass.load(strobj_str)

                    elif '.slzip.' in fname:
                        klass, obj_name = cls.class_from_fname(fname, '.slzip.')
                        buffer = io.BytesIO(obj_stream.read())
                        buffer.seek(0)
                        obj = klass.load(buffer)

                    elif fname.endswith('.lstzip'):
                        obj_name = fname[:-7]
                        buffer = io.BytesIO(obj_stream.read())
                        buffer.seek(0)
                        obj = cls._load(buffer)
                        del buffer
                        obj = [obj[str(i)] for i in range(len(obj))]

                    else:
                        try:
                            obj = json.load(obj_stream)
                            obj_name = fname.split('.')[0]
                        except TypeError:
                            warnings.warn('"{}" failed to be read as json'.format(fname))
                            try:
                                obj = pickle.dump(obj_stream)
                                obj_name = fname.split('.')[0]
                            except pickle.PickleError:
                                warnings.warn('"{}" failed to be read as pkl'.format(fname))

                    if obj_name is None:
                        msg = 'No protocol for reading "{}"'.format(file_info.filename)
                        if issubclass(handle_unknown, Exception):
                            raise handle_unknown(msg)
                        else:
                            handle_unknown(msg)

                data[obj_name] = obj
        return data

    def save(self, file, **kwargs):
        data = self.to_dict()
        assert isinstance(data, OrderedDict)
        if isinstance(file, str):
            with open(file, 'wb') as fp:
                self._save(fp, data, **kwargs)
        else:
            self._save(file, data, **kwargs)

    @classmethod
    def load(cls, file, data_overrides: dict=None, **kwargs):
        if isinstance(file, str):
            with open(file, 'rb') as fp:
                data = cls._load(fp, **kwargs)
        else:
            data = cls._load(file, **kwargs)

        if data_overrides:
            data.update(data_overrides)
        return cls.from_dict(**data)

    @abstractclassmethod
    def from_dict(cls, *args, **kwargs):
        pass

    @abstractmethod
    def to_dict(self) -> OrderedDict:
        pass


class TupperWare(OrderedDict, SaveLoadZipFilemixin):
    """
    A place to store stuff before putting it in the fridge
    """
    def to_dict(self) -> OrderedDict:
        return self

    @classmethod
    def from_dict(cls, *args, **kwargs):
        return cls.make(*args, **kwargs)

    def bootstrap(self):
        from .menu import Choices
        c = Choices('TupperWare actions', 'Container for keeping lasagnas fresh!',
                    p='print')
        args = c.parse_args()

        if args.p:
            pprint(self)


    def __init__(self, *args, **kwargs):
        if '_make_called' not in kwargs:
            raise ValueError('Never call TupperWare directly,'
                             'use TupperWare.make')
        else:
            kwargs.pop('_make_called')
        super().__init__(*args, **kwargs)

    @staticmethod
    def _getter(key, self):
        return self[key]

    @staticmethod
    def _setter(key, self, value):
        self[key] = value

    def __setitem__(self, key, value):
        if key not in self or not hasattr(self, key):
            setattr(self.__class__, key, property(partial(self._getter, key),
                                        partial(self._setter, key)))
        super().__setitem__(key, value)

    @classmethod
    def make(cls, *args, **kwargs):
        """
        Make a new instance of Options.
        Always use this method from instance creation
        """

        # noinspection PyShadowingNames
        class TupperWare(cls):
            pass
        kwargs['_make_called'] = True
        return TupperWare(*args, **kwargs)


class BaseFridge(SaveLoadZipFilemixin, ClassDescriptionMixin):
    def __init__(self, opt,
                 oven_cls: type,
                 recipe_cls: type,
                 cook_cls: type,
                 **boxes):
        self.shelves = defaultdict(TupperWare.make)
        for box_label, box in boxes.items():
            if box_label.endswith('_box'):
                owner = box_label[:-4]
                self.shelves[owner] = box
        self.opt = opt
        self.oven = oven_cls(opt)
        self.recipe = recipe_cls(opt)
        self.cook = cook_cls(opt, self.oven, self.recipe, self)

    def to_dict(self) -> OrderedDict:
        data = OrderedDict([('fridge', self.class_descr()),
                            ('opt', self.opt),
                            ('oven_cls', self.oven),
                            ('recipe_cls', self.recipe),
                            ('cook_cls', self.cook)
                            ])

        for owner, box in self.shelves.items():
            box_label = owner + '_box'
            data[box_label] = box
        return data

    @classmethod
    def from_dict(cls, *args, **kwargs):
        kwargs.pop('fridge', None)

        return cls(**kwargs)

    def bootstrap(self):
        pass


class UniversalFridgeLoader(SaveLoadZipFilemixin):
    def __init__(self):
        raise NotImplementedError('This class should not be instantiated!')

    def to_dict(self) -> OrderedDict:
        raise NotImplementedError('This class can only load!')

    @classmethod
    def from_dict(cls, *args, **kwargs):
        fridge_descr = ClassDescription(*kwargs.pop('fridge'))
        klass = cls.class_from_dscr(fridge_descr)
        assert issubclass(klass, BaseFridge)
        return klass.from_dict(*args, **kwargs)

    def bootstrap(self):
        pass
