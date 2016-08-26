"""
For storing lasagnas!
"""
# builtins
import re
from collections import (namedtuple, OrderedDict, defaultdict)
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from abc import (ABC, abstractmethod, abstractclassmethod)
import json
import os
import logging
from pprint import pprint
from zipfile import ZipFile, ZipInfo, ZIP_BZIP2, ZIP_STORED, ZipExtFile
import pickle
import warnings
import sys
import io
from typing import Union

# pip packages
from theano.compile import SharedVariable
from numpy import ndarray


# github packages
from elymetaclasses.abc import io as ioabc

# relative
from .utils import any_to_char_stream


class SaveLoadBase(ABC):
    @abstractclassmethod
    def from_dict(cls, *args, **kwargs) -> object:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractclassmethod
    def load(cls, file: Union[ioabc.InputStream, str]) -> object:
        pass

    @abstractmethod
    def save(self, file: Union[ioabc.InputStream, str]):
        pass

class JsonSaveLoadMixin(SaveLoadBase):
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
        try:
            return getattr(module, descr.clsname)
        except AttributeError as e:
            raise ImportError(e) from e

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


class SynthesizerMixin(ClassSaveLoadMixin):
    def to_dict(self):
        bases = self.__class__.__bases__
        if len([b for b in bases if issubclass(b, SynthesizerMixin)]) > 1:
            raise NotImplementedError('Synthesizing already synthesized bases'
                                      'is currently not supported.\nbases: ' + str(bases))
        return {'base_descriptions': [self.dscr_from_class(klass)._asdict()
                                      for klass in bases],
                'metaclass_description': self.dscr_from_class(type(self.__class__))._asdict(),
                'namespace': {'__module__': self.__class__.__module__,
                              '__qualname__': self.__class__.__qualname__},
                'name': self.__class__.__name__,
                'synth_descr': ClassSaveLoadMixin.to_dict(self)}

    @classmethod
    def from_dict(cls, synth_descr, metaclass_description, name,
                  base_descriptions, namespace):
        synth_descr = ClassDescription(**synth_descr)
        try:
            synth_cls = cls.class_from_dscr(synth_descr)
            return synth_cls
        except ImportError:
            pass

        metaclass = cls.class_from_dscr(ClassDescription(**metaclass_description))
        bases = tuple(cls.class_from_dscr(ClassDescription(**descr))
                      for descr in base_descriptions)
        synth_cls = metaclass.__new__(metaclass, name, bases, namespace)
        if synth_descr.modulename not in sys.modules:
            sys.modules[synth_descr.modulename] = type(sys)(synth_descr.modulename)

        setattr(sys.modules[synth_descr.modulename], synth_descr.clsname, synth_cls)
        return synth_cls


class ZipSaveError(Exception):
    pass

class ZipLoadError(Exception):
    pass


class SaveLoadZipFilemixin(ClassDescriptionMixin, SaveLoadBase):
    pickle_classes = [SharedVariable,
                      ndarray]
    ignore_extentions = ['.py', '.pyc']
    @property
    def _pickle_classes(self):
        return tuple(self.pickle_classes)

    def bootstrap(self):
        pass

    def _save(self, stream, data, manifest, compression=ZIP_STORED,
              handle_unknown=warnings.warn):
        with ZipFile(stream, mode='w', compression=compression) as zf:
            for obj_name, obj in data.items():
                ext = None
                buffer = io.BytesIO()
                if isinstance(obj, ClassSaveLoadMixin):
                    if isinstance(obj, SynthesizerMixin):
                        ext = 'slsyn'
                    else:
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

                elif isinstance(obj, ndarray):
                    import numpy
                    ext = 'npy'
                    numpy.save(buffer, obj)

                elif isinstance(obj, self._pickle_classes):
                    ext = 'pkl'
                    pickle.dump(obj, buffer)

                elif isinstance(obj, Sequence) and obj and isinstance(obj[0], self._pickle_classes):
                    ext = 'lstzip'
                    self._save(buffer,
                               dict((str(i), o) for i, o in enumerate(obj)),
                               tuple(),
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
            self._write_manifest(zf, manifest)

    @classmethod
    def _load(cls, stream, handle_unknown=warnings.warn, ignores=tuple()):
        data = OrderedDict()
        with ZipFile(stream, mode='r') as zf:
            for file_info in zf.filelist:
                fname = file_info.filename
                if fname in ignores:
                    continue
                if '/' in fname or any(fname.endswith(ign) for ign in
                                           cls.ignore_extentions):
                        continue
                assert isinstance(file_info, ZipInfo)
                obj = None
                obj_name = None
                with zf.open(file_info, mode='r') as obj_stream:
                    if fname.endswith('.pkl'):
                        obj_name = fname[:-4]
                        obj = pickle.load(obj_stream)

                    elif fname.endswith('.json'):
                        obj_name = fname[:-5]
                        obj = json.loads(obj_stream.read().decode('utf8'))

                    elif fname.endswith('.npy'):
                        import numpy
                        obj_name = fname[:-4]
                        seekable_stream = io.BytesIO(obj_stream.read())
                        seekable_stream.seek(0)
                        obj = numpy.load(seekable_stream)
                        del seekable_stream

                    elif '.slsyn' in fname:
                        strobj_str = io.TextIOWrapper(obj_stream)
                        obj = SynthesizerMixin.load(strobj_str)
                        obj_name = fname[:-6]

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
                        obj = klass.load(buffer, ignores=ignores)

                    elif fname.endswith('.lstzip'):
                        obj_name = fname[:-7]
                        buffer = io.BytesIO(obj_stream.read())
                        buffer.seek(0)
                        obj = cls._load(buffer, ignores=ignores)
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

    def _standardize_modules(self, module):
        if isinstance(module, str):
            module = import_module(module)
            return self._standardize_modules(module)

        if isinstance(module, type(sys)):
            return module, module.__name__

        if isinstance(module, (tuple, list)):
            return self._standardize_modules(module[0])[0], module[1]

    def _write_manifest(self, main_zf, manifest: Sequence):
        module_names = set()
        for module, module_name in (self._standardize_modules(m) for m in manifest):
            module_names.add(module_name)
            self._zip_module(main_zf, module, module_name)

        try:
            main_zf.getinfo('__main__.py')
        except KeyError:
            from . import bootstrap
            self._zip_module(main_zf, bootstrap, '__main__')
            module_names.add('__main__')
        return module_names

    def _zip_module(self, main_zf: ZipFile, module, module_name):
        folder, pyfile = os.path.split(module.__file__)
        zipped = not os.path.isfile(module.__file__)
        if module.__package__ == module.__name__:
            package_name = module_name
        else:
            package_name = ""

        if zipped:
            self._zip_module_in_archive(main_zf, package_name, folder,
                                        pyfile, module_name)
        else:
            self._zip_module_in_filesystem(main_zf, package_name, folder,
                                           pyfile, module_name)

    def _zip_module_in_archive(self, main_zf: ZipFile, package_name, folder,
                               pyfile, name):
        """
            zf_path = re.findall(r'(.+\.lfr)' + os.sep +    # main zip
                         r'(.+' + os.sep + r')?' +  # folder in zip (may not be present)
                         r'(.+?$)',                 # .py file
                         filename)
        """
        print(package_name)
        subfolders = folder.split(os.sep)
        zip_path = subfolders.pop(0)
        while not os.path.isfile(zip_path):
            zip_path += os.sep + subfolders.pop(0)

        with open(zip_path, 'rb') as fp:
            with ZipFile(fp, 'r') as zf:
                # single file module
                if not package_name:
                    path_in_zip = os.sep.join(subfolders + [pyfile])
                    file_info = zf.getinfo(path_in_zip)
                    with zf.open(file_info, 'r') as zf_mod:
                        main_zf.writestr(name + '.py', zf_mod.read())
                        return

                for file_info in zf.filelist:
                    package_folder = os.sep.join(subfolders)
                    print(file_info)
                    l = len(package_folder)
                    if file_info.filename.startswith(package_folder):
                        path_in_package = file_info.filename[l:]
                        file_info.filename = package_name + path_in_package
                        with zf.open(file_info, 'r') as zff:
                            print('writing')
                            main_zf.writestr(file_info, zff.read())
                return

    def _zip_module_in_filesystem(self, main_zf: ZipFile, package_name, folder,
                                  pyfile, name):
        # single file module
        if not package_name:
            file_path = folder + os.sep + pyfile if folder else pyfile
            main_zf.write(file_path, arcname=name + '.py')
            return

        l = len(folder)
        stack = list(os.scandir(folder))
        while stack:
            file = stack.pop()
            if file.is_file():
                path_in_package = file.path[l:]
                main_zf.write(file.path, package_name + path_in_package)
            elif file.is_dir():
                stack.extend(os.scandir(file.path))

    def save(self, file, manifest=tuple(), **kwargs):
        data = self.to_dict()
        assert isinstance(data, OrderedDict)
        if isinstance(file, str):
            if manifest:    # when including manifest we might need to buffer
                fp_buf = io.BytesIO()
                self._save(fp_buf, data, manifest, **kwargs)
                with open(file, 'wb') as fp:
                    fp.write(fp_buf.getvalue())
            else:
                with open(file, 'wb') as fp:
                    self._save(fp, data, manifest, **kwargs)
        else:
            self._save(file, data, manifest, **kwargs)

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

        self.box = self.shelves['fridge']
        self.opt = opt
        self.oven = oven_cls(opt)
        self.recipe = recipe_cls(opt)
        self.cook = cook_cls(opt, self.oven, self.recipe, self)

    def to_dict(self) -> OrderedDict:
        self.cook.close_all_shops()
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


# noinspection PyUnresolvedReferences
class BaseBootstrapMixin:
    assume_exists = ['lasagnecaterer']

    class BootstrapWrapper(ClassSaveLoadMixin):
        def __init__(self, boot_func):
            self.boot_func = boot_func

        def __call__(self, *args, **kwargs):
            self.boot_func(*args, **kwargs)

        def to_dict(self):
            descr = self.dscr_from_class(self.boot_func)
            return descr._asdict()

    @property
    def has_bootstrap_func(self):
        return 'bootstrap_func' in self.box

    def _wrap_boot_func(self):
        if not isinstance(self.box['bootstrap_func'], SaveLoadBase):
            self.box['bootstrap_func'] = self.BootstrapWrapper(self.box['bootstrap_func'])

    def needed_modules(self):
        if isinstance(self.box['bootstrap_func'], self.BootstrapWrapper):
            bootstrap_func = self.box['bootstrap_func'].boot_func
        else:
            bootstrap_func = self.box['bootstrap_func']

        needed = list()

        module, module_name = self._standardize_modules(bootstrap_func.__module__)
        if module.__package__ or module_name not in self.assume_exists:
            needed.append(module_name)

        return needed + self.box.get('aux_modules', [])

    def bootstrap(self):
        if self.has_bootstrap_func:
            self.box['bootstrap_func'](self)

    def set_bootstrap(self, bootstrap_func, *aux_modules):
        self.box['bootstrap_func'] = bootstrap_func
        self.box['aux_modules'] = [self._standardize_modules(m)[1]
                                       for m in aux_modules]

    def to_dict(self):
        self._wrap_boot_func()
        return super().to_dict()

    def _write_manifest(self, main_zf, manifest: Sequence):
        module_names = super()._write_manifest(main_zf, manifest)
        if not self.has_bootstrap_func:
            return module_names

        need_mods = [m for m in self.needed_modules() if m not in module_names]

        return module_names.update(super()._write_manifest(main_zf, need_mods))


class BootstrapFridge(BaseBootstrapMixin, BaseFridge):
    pass