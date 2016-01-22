"""
For storing lasagnas!
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

class ZipSaveError(Exception):
    pass

class ZipLoadError(Exception):
    pass

class SaveLoadZipFilemixin:
    pickle_classes = [SharedVariable,
                      ndarray]

    @property
    def _pickle_classes(self):
        return tuple(self.pickle_classes)

    @staticmethod
    def dscr_from_class(cls: type):
        module = sys.modules[cls.__module__]
        modulename = module.__name__
        clsname = cls.__name__
        return modulename, clsname

    @staticmethod
    def class_from_dsrc(fname, splitter):
        obj_name, ending = fname.split(splitter)
        modulename = '.'.join(ending.split('.')[:-1])
        clsname = ending.split('.')[-1]
        module = sys.modules[modulename]
        klass = getattr(module, clsname)
        return klass, obj_name

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
                if isinstance(obj, JsonSaveLoadMixin):
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

                    elif '.sljson.' in fname:
                        klass, obj_name = cls.class_from_dsrc(fname, '.sljson.')
                        strobj_str = io.TextIOWrapper(obj_stream)
                        obj = klass.load(strobj_str)

                    elif '.slzip.' in fname:
                        klass, obj_name = cls.class_from_dsrc(fname, '.slzip.')
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
    def load(cls, file, **kwargs):
        if isinstance(file, str):
            with open(file, 'rb') as fp:
                data = cls._load(fp, **kwargs)
        else:
            data = cls._load(file, **kwargs)

        return cls.from_dict(**data)

    @abstractclassmethod
    def from_dict(cls, *args, **kwargs):
        pass

    @abstractmethod
    def to_dict(self) -> OrderedDict:
        pass