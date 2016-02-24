# builtins
import _pyio
import inspect
import io
import _pyio  # for subclassing
import os
import random
from collections import namedtuple, OrderedDict, deque, Generator, defaultdict, \
    MutableSequence, UserDict
from itertools import chain
from operator import attrgetter, itemgetter
from typing import Union, List, Any
import re
import sys
import asyncio
import warnings
from weakref import WeakKeyDictionary

from decorator import contextmanager
from functools import singledispatch, partial, wraps
from subprocess import Popen, PIPE
import time
import asyncio.subprocess as aiosubprocess

# pip
try:
    # MUST BE IMPORTED BEFORE BOKEH
    from tornado.platform.asyncio import AsyncIOMainLoop

    AsyncIOMainLoop().install()
    # MUST BE IMPORTED BEFORE BOKEH
except AssertionError:
    warnings.warn('Could not swap out tornado AIO-loop')

import numpy as np
from progressbar import (ProgressBar, Percentage, SimpleProgress, Bar, Widget)
from bokeh.models import ColumnDataSource, Range1d, FactorRange, HoverTool, \
    TextEditor
from bokeh.plotting import figure, hplot, vplot, gridplot
from bokeh.client import push_session
from bokeh.io import curdoc, curstate
from bokeh.palettes import RdYlGn5, RdYlGn7
from bokeh.models.widgets.tables import DataTable

# github
from elymetaclasses.annotations import type_assert
from elymetaclasses.abc import io as ioabc

all = ['from_start', 'any_to_stream', 'seekable']


@contextmanager
@type_assert(ioabc.SeekableInputStream)
def from_start(inp):
    pos = inp.tell()
    inp.seek(0)
    yield inp
    inp.seek(pos)


def stream_to_buffer(inp: ioabc.InputStream):
    if 'b' in inp.mode:
        return io.BytesIO(inp.read())
    return io.StringIO(inp.read())


def seekable(inp: ioabc.InputStream):
    if not isinstance(inp, ioabc.SeekableStream):
        return False

    if hasattr(inp, 'seekable'):
        return inp.seekable

    try:
        inp.seek(0, 1)
        return True
    except io.UnsupportedOperation:
        pass


def force_seek(inp: ioabc.InputStream) -> ioabc.SeekableInputStream:
    if seekable(inp):
        return inp
    return stream_to_buffer(inp)


@singledispatch
def any_to_char_stream(inp: Union[str, ioabc.InputStream, bytes],
                       output_type: Union[str, bytes] = str,
                       force_seekable=False) -> Union[ioabc.InputStream,
                                                      ioabc.SeekableInputStream]:
    """
    Transfrom any kind of input into a
    :param inp:
        something that can be turned into an input stream:
            str: filename -> opened file_descriptor
                else -> BufferStream (StringIO or BytesIO)

            input_stream: -> passthrough, possibly wrapped to match output_type
            bytes -> BufferStream (StringIO or BytesIO)
    :param output_type: type
        the kind of data that the stream should return
    :param force_seekable: force returned stream to be seekable. may require
                           buffering a large file!
    :return: InputStream
    """
    pass


@any_to_char_stream.register(str)
def str_to_stream(inp: str, output_type: Union[str, bytes] = str,
                  force_seekable=False):
    if output_type not in (str, bytes):
        raise ValueError('output_type must be str or bytes')

    if os.path.isfile(inp):
        if output_type is str:
            return open(inp, 'r')
        return open(inp, 'rb')

    if output_type is str:
        return io.StringIO(inp)
    return io.BytesIO(inp.encode())


@any_to_char_stream.register(ioabc.InputStream)
def stream_to_stream(inp: ioabc.InputStream,
                     output_type: Union[str, bytes] = str,
                     force_seekable=False):
    if output_type not in (str, bytes):
        raise ValueError('output_type must be str or bytes')

    if hasattr(inp, 'mode'):
        mode = inp.mode
    elif isinstance(inp, (io.BytesIO, _pyio.BytesIO)):
        mode = 'b'
    elif isinstance(inp, (io.StringIO, _pyio.StringIO)):
        mode = ''
    else:
        # giving up
        return any_to_char_stream(inp.read())

    if force_seekable and not seekable(inp):
        # convert to buffered
        if 'b' in mode:
            return io.BytesIO(inp.read().encode())
        return io.StringIO(inp.read().decode())

    # try to reuse stream
    if 'b' in mode:
        if output_type is bytes:
            return inp
        return io.TextIOWrapper(inp)

    else:
        if output_type is str:
            return inp

        if isinstance(inp, (_pyio.TextIOWrapper, io.TextIOWrapper)):
            # noinspection PyUnresolvedReferences
            return inp.buffer
        return io.BytesIO(inp.read().encode())


@any_to_char_stream.register(bytes)
def bytes_to_stream(inp: bytes, output_type: Union[str, bytes] = str,
                    force_seekable=False):
    if output_type is bytes:
        return io.BytesIO(inp)
    return io.StringIO(inp.decode())


from elymetaclasses.events import ChainedPropsMetaClass
from abc import ABCMeta


class ChainPropsABCMetaclass(ChainedPropsMetaClass, ABCMeta):
    pass


class MixinProp(property):
    """
    Descriptor for use with mixins to to avoid calling __init__
    """

    def __init__(self, default=0.0, super_call=None, return_type=Any):
        self.default = default
        self.super_call = super_call
        self.return_type = return_type
        self.owners = WeakKeyDictionary()
        self.instances = WeakKeyDictionary()

    def _get_super(self, instance, obj_type):
        try:
            result = getattr(super(obj_type, instance), self.super_call)
        except AttributeError:
            parents = [klass.__qualname__ for klass in obj_type.__mro__[1:]]
            raise AttributeError('A mixin requires {0} to have an attribute'
                                 '{1} in one of the parents classes:'
                                 '\n {2}'.format(instance, parents,
                                                 self.super_call))

        if self.return_type is Any:
            return result
        try:
            assert isinstance(result, self.return_type)
            return result
        except AssertionError:
            raise AssertionError('A mixin expects return type of {0}'
                                 'to be an instance of {1}.'
                                 ' it was found to be {2}'.format(
                self.super_call,
                self.return_type,
                type(result)))

    def __get__(self, instance, obj_type):
        if self.super_call:
            return self._get_super(instance, obj_type)
        if instance not in self.instances:
            self.instances[instance] = self.default
        return self.instances[instance]

    def __set__(self, instance, value):
        if self.super_call:
            raise NotImplementedError
        self.instances[instance] = value


class MixinRequires:
    """
    def __init__(self, super_call, return_type=Any):
        self.super_call = super_call
        self.return_type = return_type
        self.owners = WeakKeyDictionary()
        self.instances = WeakKeyDictionary()
        for methodname, method in return_type.__dict__:
            setattr(self, methodname, method)

    def __get__(self, instance, owner):
        try:
            result = getattr(super(owner, instance), self.super_call)
        except AttributeError:
            parents = [klass.__qualname__ for klass in owner.__mro__[1:]]
            raise AttributeError('A mixin requires {0} to have an attribute'
                                 '{1} in one of the parents classes:'
                                 '\n {2}'.format(instance, parents,
                                                 self.super_call))

        if self.return_type is Any:
            return result
        try:
            assert isinstance(result, self.return_type)
            return result
        except AssertionError:
            raise AssertionError('A mixin expects return type of {0}'
                                 'to be an instance of {1}.'
                                 ' it was found to be {2}'.format(
                self.super_call,
                self.return_type,
                type(result)))
    """

    @classmethod
    def mixin_decorator(cls, mixin_cls):
        remove = list()
        for name, attr in mixin_cls.__dict__.items():
            if isinstance(attr, type):
                if issubclass(attr, cls):
                    remove.append(name)

        for remname in remove:
            delattr(mixin_cls, remname)
        return mixin_cls

    @classmethod
    def mock(cls, mock_classes, _=None):
        try:
            assert isinstance(_, mock_classes)
            return _
        except AssertionError:
            return cls


def mixin_mock(obj):
    if isinstance(obj, property):
        return obj.setter('mocked')

    else:
        remove = list()
        for name, attr in obj.__dict__.items():
            if isinstance(attr, property) and attr.fset == 'mocked':
                remove.append(name)

        for remname in remove:
            delattr(obj, remname)
        return obj


class Message(Widget):
    'Returns progress as a count of the total (e.g.: "5 of 47")'

    __slots__ = ('message', 'fmt', 'max_width')

    def __init__(self, message, max_width=None):
        self.message = message
        self.max_width = max(max_width or len(message), 1)
        self.fmt = ' {:' + str(self.max_width) + '} '

    def update(self, pbar):
        return self.fmt.format(self.message[:self.max_width])


def pbar(what, max_val, *args):
    msg = Message(*args)
    return ProgressBar(max_val, fd=sys.stdout,
                       widgets=[SimpleProgress(), ' ',
                                what, msg,
                                Percentage(), Bar(), '\n']), msg


FNULL = open(os.devnull, 'w')


class AsyncProcWrapper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._p = None

    async def start(self):
        self._p = await asyncio.create_subprocess_exec(*self.args,
                                                       **self.kwargs)

    @property
    def p(self) -> aiosubprocess.Process:
        return self._p

    @property
    def stdin(self) -> aiosubprocess.streams.StreamWriter:
        return self.p.stdin

    @property
    def stdout(self) -> aiosubprocess.streams.StreamReader:
        return self.p.stdout

    @property
    def stderr(self) -> aiosubprocess.streams.StreamReader:
        return self.p.stderr

    @classmethod
    async def create(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        await obj.start()
        return obj

    async def communicate(self) -> List[bytes]:
        err = await self.stderr.read()
        out = await self.stdout.read()
        return out, err


async def async_subprocess(*args, **kwargs):
    return await AsyncProcWrapper.create(*args, **kwargs)


class AsyncBytesIO(asyncio.streams.StreamReader, _pyio.BytesIO):
    def __init__(self):
        super().__init__()
        self._pos = 0
        self._alt_out = None

    def write(self, s):
        b = s.encode() if isinstance(s, str) else s
        self.feed_data(b)
        if self._alt_out:
            self._alt_out.write(s)

    @contextmanager
    def redirect_stdout(self, copy=False):
        stdout, sys.stdout = sys.stdout, self  # io.TextIOWrapper(self)
        if copy:
            self._alt_out = stdout
        yield
        sys.stdout = stdout

    async def read_all(self):
        await self._wait_for_data('updater')
        b = self.getvalue()
        self._buffer.clear()
        return b


class BokehConsole:
    def __init__(self, n=10, max_line_len=200, input_bottom=True):
        self.n = n
        self.max_line_len = max_line_len
        self.source = self.make_source()
        self.p = self.make_plot()
        self.line_buffer = deque(self.source.data['text'])
        self._rotate = 1 if not input_bottom else -1
        self._pos = 0 if not input_bottom else -1
        super(BokehConsole, self).__init__()

    def make_source(self):
        return ColumnDataSource({'text': [''] * self.n,
                                 'zeros': [0] * self.n,
                                 'line': list(reversed(range(self.n)))})

    def make_plot(self):
        p = figure(y_range=(Range1d(-1, self.n + 1)),
                   x_range=(Range1d(-2, self.max_line_len + 1)), tools='hover',
                   width=int(self.max_line_len * 6.35 + 160),
                   height=(self.n + 2) * 16 + 100)
        p.text('zeros', 'line', 'text', source=self.source)
        p.axis.visible = None
        p.toolbar_location = 'below'
        g = p.grid
        g.grid_line_color = '#FFFFFF'
        return p

    def _push_line(self, line):
        self.line_buffer.rotate(self._rotate)
        self.line_buffer[self._pos] = line
        self.source.data['text'] = list(self.line_buffer)

    def _push_lines(self, lines):
        l = len(lines)
        if self._pos == -1:  # lines come in from bottom
            for i, line in enumerate(lines):
                self.line_buffer[i] = line
            self.line_buffer.rotate(self._rotate * l)
        else:  # lines come in from top
            self.line_buffer.rotate(self._rotate * l)
            for i, line in enumerate(lines):
                self.line_buffer[i] = line
        self.source.data['text'] = list(self.line_buffer)

    def output_text(self, s):
        lines = list()
        for line in s.split('\n'):
            if not line:
                continue
            if len(line) <= self.max_line_len:
                lines.append(line)
            else:
                tokens = list()
                i = -1
                for token in line.split(' '):
                    i += 1 + len(token)
                    if i > self.max_line_len:
                        lines.append(' '.join(tokens))
                        tokens = [token]
                        i = len(token)
                    else:
                        tokens.append(token)
                lines.append(' '.join(tokens))
        self._push_lines(lines)


JobProgress = namedtuple('JobProgress', 'name percent state')
Command = namedtuple('Command', 'command args kwargs')



BytesStdOut = namedtuple('BytesStdOut', 'bytes')
TextStdOut = namedtuple('TextStdOut', 'text')
BytesStdErr = namedtuple('BytesStdErr', 'bytes')
TextStdErr = namedtuple('TextStdErr', 'text')


class ChangeStream(asyncio.Queue):
    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop = loop or asyncio.get_event_loop()
        self.sentinel2subscribers = defaultdict(list)
        notifier_init = partial(self.ChangeNotifier, self._loop)
        self.sentinel2notifier = defaultdict(notifier_init)
        self._terminating = False
        self.terminated = False

    class ChangeNotifier:
        def __init__(self, loop):
            """
            :param notify_classes: Change classes to trigger a wake-up
            :param loop:
            :return:
            """
            self._waiters = list()
            self._loop = loop
            self.change = None

        def __len__(self):
            return len(self._waiters)

        def __bool__(self):
            return bool(self._waiters)

        def wake_up(self, change):
            self.change = change
            self._wakeup_waiters()

        def _wakeup_waiters(self):
            waiters = self._waiters
            if waiters:
                for waiter in waiters:
                    if not waiter.cancelled():
                        waiter.set_result(None)
            waiters.clear()

        async def wait_for_change(self):
            waiter = asyncio.futures.Future(loop=self._loop)
            self._waiters.append(waiter)
            try:
                await waiter
                return self.change
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

    class BytesRedirecter:
        def __init__(self, change_stream: asyncio.Queue, copy_out,
                     wrap_class=BytesStdOut):
            self.change_stream = change_stream
            self._copy_out = copy_out
            self.wrap_class = wrap_class
            self.put_bytes = True if hasattr(wrap_class, 'bytes') else False

        def write(self, b):
            is_bytes = isinstance(b, bytes)
            # no change needed
            if (is_bytes and self.put_bytes) or (
                        not is_bytes and not self.put_bytes):
                self.change_stream.put_nowait(self.wrap_class(b))

            # is bytes, but should be str
            elif is_bytes:
                self.change_stream.put_nowait(self.wrap_class(b.decode()))

            # is not bytes, but should be
            else:
                self.change_stream.put_nowait(self.wrap_class(b.encode()))

            if self._copy_out:
                self._copy_out.write(b)

        def flush(self):
            if self._copy_out:
                self._copy_out.flush()

    class TextRedirecter:
        def __init__(self, change_stream: asyncio.Queue, copy_out,
                     wrap_class=TextStdOut):
            self.change_stream = change_stream
            self._copy_out = copy_out
            self.wrap_class = wrap_class
            self.put_str = True if hasattr(wrap_class, 'text') else False

        def write(self, s):
            is_text = isinstance(s, str)
            # no change needed
            if (is_text and self.put_str) or (not is_text and not self.put_str):
                self.change_stream.put_nowait(self.wrap_class(s))

            # is bytes, but should be str
            elif not is_text:
                self.change_stream.put_nowait(self.wrap_class(s.decode()))

            # is not bytes, but should be
            else:
                self.change_stream.put_nowait(self.wrap_class(s.encode()))

            if self._copy_out:
                self._copy_out.write(s)

        def flush(self):
            if self._copy_out:
                self._copy_out.flush()

    @contextmanager
    def redirect_stdout(self, copy=False, out_type=str, **kwargs):
        if copy:
            copy_out = sys.stdout
        else:
            copy_out = None

        if out_type is str:
            redirecter = self.TextRedirecter(self, copy_out, **kwargs)
        else:
            redirecter = self.BytesRedirecter(self, copy_out, **kwargs)

        stdout, sys.stdout = sys.stdout, redirecter  # io.TextIOWrapper(self)
        yield
        sys.stdout = stdout

    @contextmanager
    def redirect_stderr(self, copy=False, out_type=str, **kwargs):
        if copy:
            copy_out = sys.stdout
        else:
            copy_out = None

        if out_type is str:
            if 'wrap_class' not in kwargs:
                kwargs['wrap_class'] = TextStdErr
            redirecter = self.TextRedirecter(self, copy_out, **kwargs)
        else:
            if 'wrap_class' not in kwargs:
                kwargs['wrap_class'] = BytesStdErr
            redirecter = self.BytesRedirecter(self, copy_out, **kwargs)

        stdout, sys.stderr = sys.stderr, redirecter  # io.TextIOWrapper(self)
        yield
        sys.stderr = stdout

    async def wait_for_change(self, *sentinels):
        sentinel = tuple(sorted(sentinels))
        return await self.sentinel2notifier[sentinel].wait_for_change()

    def register_subscriber(self, waiter_coro):
        if inspect.isfunction(waiter_coro):
            waiter_coro = waiter_coro()
        sentinel = next(waiter_coro)
        self.sentinel2subscribers[sentinel].append(waiter_coro)

    async def start(self):
        if self.terminated:
            warnings.warn('Trying to start a terminated {}'.format(self.__class__))
            return

        while True:
            change = await self.get()
            a = 'hej'
            for sentinel, waiters in self.sentinel2subscribers.items():
                if isinstance(change, sentinel):
                    dead_waiters = list()
                    for waiter in waiters:
                        try:
                            waiter.send(change)
                        except StopIteration:
                            dead_waiters.append(waiter)
                    for waiter in dead_waiters:
                        waiters.remove(waiter)

            for sentinel, notifier in self.sentinel2notifier.items():
                if isinstance(change, sentinel):
                    notifier.wake_up(change)

            if isinstance(change, Command):
                if change.command == 'terminate':
                    self._terminating = False
                    self.terminated = True
                    while not self.empty():
                        self.get_nowait()
                    self._unfinished_tasks = 0
                    self.sentinel2subscribers.clear()
                    return

    async def terminate(self):
        if self.terminated:
            return
        if not self._terminating:
            self._terminating = True
            await self.put(Command('terminate', (), ()))

        while self._terminating:
            await self.wait_for_change(Command)
        assert self.terminated


@contextmanager
def redirect_to_changestream(change_stream: asyncio.Queue, err=False):
    stdout = sys.stdout

# Unless explicitly defined as Nvidia device all GPUs are considered as cuda
# devices
CPU = namedtuple('CPU', 'dev load')
GPU = namedtuple('GPU', 'dev free nvdev')
GPUNv = namedtuple('GPUNv', 'nvdev free load')  # <- nvidia device
GPUComb = namedtuple('GPUComb', 'dev free load')

GPUProcess = namedtuple('GPUProcess', 'pid owner dev memusage')
GPUNvProcess = namedtuple('GPUNvProcess',
                          'nvdev pid memusage')  # <- nvidia device

nvgpu_used_tot_load = re.compile('(\d+)MiB / (\d+)MiB \|\W*(\d+)%')
nvgpu_nvdev = re.compile('\|\W*(\d+)[^\|]+\|\W*[\dA-F]+:[\dA-F]+:[\dA-F]+\.')
nvgpu_divider = re.compile('^\+(-+\+)+\n?$')

gpu_dev_nvdev_used_tot = re.compile(
    'Device  (\d).*nvidia-smi\W*(\d+).+ (\d+) of (\d+) MiB Used')

gpu_nvdev_pid_mem = re.compile('(\d+)\W*(\d+).+?(\d+)MiB \|\n?$')

cpu_dev_load = re.compile('%Cpu(\d+)\W*:.*?(\d+)\[')
cpu_dev_us_sy = re.compile('%Cpu(\d+)\W*:\W* (\d+\.\d+) us,\W+(\d+\.\d+)')


def nv_line2nvdev(line, prev_nvdev=None):
    if nvgpu_divider.match(line):
        return None

    res = nvgpu_nvdev.findall(line)
    return int(res[0]) if res else prev_nvdev


def nv_line2GPUNv(line, nvdev):
    res = nvgpu_used_tot_load.findall(line)
    if res:
        if nvdev is None:
            raise ValueError('Found a valid info line, but i have no device')
        used, tot, load = res[0]
        free = int(tot) - int(used)
        return GPUNv(nvdev, free, float(load))


def nv_line2GPUNvProcess(line):
    res = gpu_nvdev_pid_mem.findall(line)
    if not res:
        return None
    return GPUNvProcess(*(int(r) for r in res[0]))


def cuda_line2GPU(line) -> GPU:
    res = gpu_dev_nvdev_used_tot.findall(line)
    if not res:
        return None
    dev, nvdev, used, tot = res[0]
    free = int(tot) - int(used)
    return GPU(int(dev), free, int(nvdev))


def lines2CPUs(lines):
    if not isinstance(lines, str):
        lines = '\n'.join(lines)
    res = cpu_dev_load.findall(lines)
    if res:
        return [CPU(int(dev), float(load)) for dev, load in res]

    res = cpu_dev_us_sy.findall(lines)
    if res:
        return [CPU(int(dev), float(us) + float(sy)) for dev, us, sy in res]
    return []


class RessourceMonitor:
    def __init__(self, change_stream: ChangeStream,
                 async_exec=async_subprocess,
                 normal_exec=Popen):
        # self.sources = self.init_sources()
        self.async_exec = async_exec
        self.normal_exec = normal_exec
        self.change_stream = change_stream
        self.monitors = OrderedDict()
        self.terminated = False
        self._terminating = True

    def init_sources(self):
        _gpus = self.gpus_mem()
        _cpus = self.cpus()

        source = dict(cpu=ColumnDataSource({
            'cpu_dev': [gpu.dev for gpu in _cpus],
            'cpu_load': [gpu.load for gpu in _cpus]}),
            gpumem=ColumnDataSource({
                'gpu_dev': [gpu.dev for gpu in _gpus],
                'gpu_free': [gpu.free for gpu in _gpus],
            }),
            gpuclock=ColumnDataSource({
                'gpu_dev': [gpu.dev for gpu in _gpus],
                'gpu_load': [gpu.free for gpu in _gpus],
            }))
        return source

    async def terminate(self):
        if self.terminated:
            return
        if not self._terminating:
            self._terminating = True

        while self.monitors:
            await self.change_stream.wait_for_change(chain(*self.monitors.values()))
        self._terminating, self.terminated = False, True


    @contextmanager
    def register_mon(self, *sentinels):
        id = hash(random.random())
        self.monitors[id] = sentinels
        yield
        del self.monitors[id]

    def mon_decorator(*sentinels):
        if sentinels and isinstance(sentinels[0], asyncio.Queue):
            sentinels = sentinels[1:]
        if not sentinels:
            raise ValueError('Cannot define a monitor without a sentinel')

        def decorator(func):
            @wraps(func)
            async def wrapper(instance, *args, **kwargs):
                if instance.terminated or instance._terminating:
                    warnings.warn('Trying to start a terminated {}'.format(instance.__class__))
                    return
                with instance.register_mon(*sentinels):
                    await func(instance, *args, **kwargs)

            return wrapper
        return decorator

    @mon_decorator(CPU)
    async def cpus_mon(self):
        subproc_exec = self.async_exec
        p = await subproc_exec('top', '-b', '-p0', '-d3',
                               stdout=PIPE)
        cpus = dict()
        while not self._terminating:
            line = (await p.stdout.readline()).decode()
            cpu = lines2CPUs(line)
            if cpu:
                cpu = cpu[0]
                prev_load = cpus.get(cpu.dev, None)
                if prev_load != cpu.load:
                    cpus[cpu.dev] = cpu.load
                    await self.change_stream.put(cpu)

    @mon_decorator(GPUComb, GPUProcess)
    async def gpus_mon(self, loop: asyncio.BaseEventLoop=None,
                       ignore=tuple()):
        subproc_exec = self.async_exec
        nv2cuda, pid2owner = await asyncio.gather(
            self.nv2cuda_coro(subproc_exec),
            self.pid2owner_coro(subproc_exec))

        async def start_proc():
            return (await self.async_exec('nvidia-smi', '-l', '2',
                                          stdout=asyncio.subprocess.PIPE,
                                          stderr=FNULL))

        p = await start_proc()

        loop = loop or asyncio.get_event_loop()
        gpus = dict()
        gpu_nvprocs = dict()
        do_GPUComb = GPUComb not in ignore
        do_GPUProcess = GPUProcess not in ignore
        nvdev = None
        tasks = list()
        seen_pids = list()
        last_update = time.time()
        while not self._terminating:
            line = await p.stdout.readline()
            if p.stdout.at_eof():
                warnings.warn('nvidia-smi died..  restarting')
                p = await start_proc()
                warnings.warn('nvidia-smi restarted')
            line = line.decode()
            if do_GPUComb:
                nvdev = nv_line2nvdev(line, nvdev)
                nvgpu = nv_line2GPUNv(line, nvdev)

                # a gpu was found in stdout
                if nvgpu:
                    prev_gpu = gpus.get(nvgpu.nvdev, None)
                    # has anything changed? (also update at least ever 10 sec)
                    if prev_gpu != nvgpu[1:] or time.time() - last_update > 10:
                        last_update = time.time()

                        # translate to cuda dev and update gpus
                        gpu = GPUComb(nv2cuda[nvgpu.nvdev], *nvgpu[1:])
                        gpus[nvgpu.nvdev] = nvgpu[1:]

                        # put into change stream
                        await self.change_stream.put(gpu)
                    continue

            if do_GPUProcess:
                nvproc = nv_line2GPUNvProcess(line)
                if nvproc:
                    seen_pids.append(nvproc.pid)
                    tasks.append(
                        loop.create_task(self._nvproc2proc(subproc_exec,
                                                           nvproc,
                                                           pid2owner,
                                                           nv2cuda,
                                                           gpu_nvprocs)))
                    continue

            if tasks:
                await asyncio.wait(tasks)
                tasks.clear()

                dead_pids = set(gpu_nvprocs.keys()).difference(seen_pids)
                for dead_proc in (gpu_nvprocs[pid] for pid in dead_pids):
                    await self.change_stream.put(GPUProcess(dead_proc.pid,
                                                            pid2owner[
                                                                dead_proc.pid],
                                                            nv2cuda[
                                                                dead_proc.nvdev],
                                                            0))
                    gpu_nvprocs.pop(dead_proc.pid)
                seen_pids.clear()

    @staticmethod
    def gpus_mem(subproc_exec=Popen) -> List[GPU]:
        _gpus = list()
        for line in subproc_exec(['cuda-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            gpu = cuda_line2GPU(line)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    async def gpus_mem_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('cuda-smi',
                               stdout=asyncio.subprocess.PIPE,
                               stderr=FNULL)

        data = await p.stdout.read()
        _gpus = list()
        for line in data.decode('utf8').split('\n'):
            gpu = cuda_line2GPU(line)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    def nv_gpus_mem_load(subproc_exec=Popen) -> List[GPUNv]:
        _gpus = list()
        nvdev = None
        for line in subproc_exec(['nvidia-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            nvdev = nv_line2nvdev(line, nvdev)
            gpu = nv_line2GPUNv(line, nvdev)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    async def nv_gpus_mem_load_coro(subproc_exec=async_subprocess) -> List[
        GPUNv]:
        p = await subproc_exec(['nvidia-smi'],
                               stdout=PIPE)
        data = await p.stdout.read()
        _gpus = list()
        nvdev = None
        for line in data.decode('utf8').split('\n'):
            nvdev = nv_line2nvdev(line, nvdev)
            gpu = nv_line2GPUNv(line, nvdev)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @classmethod
    def nv2cuda(cls, subproc_exec=Popen):
        gpus = cls.gpus_mem(subproc_exec)
        return dict((gpu.nvdev, gpu.dev) for gpu in gpus)

    @classmethod
    async def nv2cuda_coro(cls, subproc_exec=async_subprocess):
        gpus = await cls.gpus_mem_coro(subproc_exec)
        return dict((gpu.nvdev, gpu.dev) for gpu in gpus)

    @classmethod
    def gpus_comb(cls, subproc_exec=Popen):
        nv2cuda = cls.nv2cuda(subproc_exec)
        nvgpus = cls.nv_gpus_mem_load(subproc_exec)
        return [GPUComb(nv2cuda[gpu.nvdev], *gpu[1:]) for gpu in nvgpus]

    @classmethod
    async def gpus_comb_coro(cls, subproc_exec=async_subprocess):
        nv2cuda, nvgpus = await asyncio.gather(cls.nv2cuda_coro(subproc_exec),
                                               cls.nv_gpus_mem_load_coro(
                                                   subproc_exec))

        return [GPUComb(nv2cuda[gpu.nvdev], *gpu[1:]) for gpu in nvgpus]

    @classmethod
    def gpu_procs(cls, subproc_exec=Popen):
        nv2cuda = cls.nv2cuda(subproc_exec)
        nvprocs = dict()
        for line in subproc_exec(['nvidia-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            proc = nv_line2GPUNvProcess(line)
            if proc:
                nvprocs[proc.pid] = proc

        procs = list()
        for nvproc, owner in zip(nvprocs.values(),
                                 cls.find_pid_owner(*nvprocs.keys(),
                                                    subproc_exec=subproc_exec)):
            procs.append(GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                                    nvproc.memusage))
        return procs

    @classmethod
    async def gpu_procs_coro(cls, subproc_exec=async_subprocess):
        nv2cuda, p = await asyncio.gather(cls.nv2cuda_coro(subproc_exec),
                                          subproc_exec(['nvidia-smi'],
                                                       stdout=PIPE))
        data = await p.stdout.read()
        nvprocs = dict()
        for line in data.decode(
                'utf8').split('\n')[:-1]:

            proc = nv_line2GPUNvProcess(line)
            if proc:
                nvprocs[proc.pid] = proc

        procs = list()
        owners = await cls.find_pid_owner_coro(nvprocs.keys(), subproc_exec)
        for nvproc, owner in zip(nvprocs.values(), owners):
            procs.append(GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                                    nvproc.memusage))
        return procs

    async def _nvproc2proc(self, subproc_exec, nvproc: GPUNvProcess,
                           pid2owner: dict, nv2cuda: dict, gpu_nvprocs: dict):
        if nvproc.pid not in pid2owner:
            owner = await self.find_pid_owner_coro(nvproc.pid,
                                                   subproc_exec=subproc_exec)
            owner = owner[0]
            pid2owner[nvproc.pid] = owner
            prev_proc = None
        else:
            owner = pid2owner[nvproc.pid]
            prev_proc = gpu_nvprocs.get(nvproc.pid, None)

        if prev_proc != nvproc:
            gpu_nvprocs[nvproc.pid] = nvproc
            proc = GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                              nvproc.memusage)
            await self.change_stream.put(proc)

    @staticmethod
    def cpus(subproc_exec=Popen):
        p = subproc_exec(['top', '-n2', '-b', '-p0'], stdout=PIPE, stderr=FNULL)
        data = p.stdout.read()
        return lines2CPUs(data.decode())

    @staticmethod
    async def cpus_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('top', '-n2', '-b',
                               stdout=asyncio.subprocess.PIPE,
                               stderr=FNULL)
        data = await p.stdout.read()
        return lines2CPUs(data.decode())


    @staticmethod
    def find_pid_owner(*pids, subproc_exec=Popen):
        pids = list(str(pid) for pid in pids)
        pid_cs = ','.join(pids)
        p = subproc_exec(['ps', '-p', pid_cs, '-o', 'pid,user', 'h'],
                         stdout=PIPE,
                         stderr=PIPE)
        data, err = p.communicate()
        if err:
            raise IOError(err)
        pid2owner = dict(re.findall('\W*(\d+) (\w+)', data.decode()))
        return [pid2owner.get(pid, None) for pid in pids]

    @staticmethod
    async def find_pid_owner_coro(*pids, subproc_exec=async_subprocess):
        pids = list(str(pid) for pid in pids)
        pid_cs = ','.join(pids)
        p = await subproc_exec('ps', '-p', pid_cs, '-o', 'pid,user', 'h',
                               stdout=PIPE,
                               stderr=PIPE)

        data, err = await p.communicate()
        if err:
            raise IOError(err)
        pid2owner = dict(re.findall('\W*(\d+) (\w+)', data.decode()))
        return [pid2owner.get(pid, None) for pid in pids]

    @staticmethod
    def pid2owner(subproc_exec=Popen):
        p = subproc_exec('ps', '-A', '-o', 'pid,user', 'h',
                         stdout=PIPE,
                         stderr=PIPE)
        data, err = p.communicate()
        if err:
            raise IOError(err)
        pid2owner = dict((int(pid), owner) for pid, owner in
                         re.findall('\W*(\d+) (\w+)', data.decode()))
        return pid2owner

    @staticmethod
    async def pid2owner_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('ps', '-A', '-o', 'pid,user', 'h',
                               stdout=PIPE,
                               stderr=PIPE)
        data, err = await p.communicate()
        if err:
            raise IOError(err)
        pid2owner = dict((int(pid), owner) for pid, owner in
                         re.findall('\W*(\d+) (\w+)', data.decode()))
        return pid2owner

class BokehPlots:
    def __init__(self, change_consumer: ChangeStream):
        self.change_consumer = change_consumer
        self.terminated = False

    @staticmethod
    def _drop_in(data, field, idx, val):
        data[field] = data[field][:idx] + [val] + data[field][idx + 1:]

    @staticmethod
    def _bar_source(names, *val_pairs):
        l = len(names)
        d = {
            'zeros': [0] * l,
            'name': [str(n) for n in names],
            'name_low': [str(n) + ':0.9' for n in names],
            'name_high': [str(n) + ':0.1' for n in names]
        }
        for val_name, values in val_pairs:
            if isinstance(values, (tuple, MutableSequence)):
                d[val_name] = values
            else:
                d[val_name] = [values] * l

        return d

    @staticmethod
    def _swapaxes(kwargs):
        kwargs['x_range'], kwargs['y_range'] = kwargs['y_range'], kwargs[
            'x_range']
        kwargs['x_axis_label'], kwargs['y_axis_label'] = kwargs['y_axis_label'], \
                                                         kwargs[
                                                             'x_axis_label']

    def cpu_bars(self, vertical=True, **kwargs):
        cpus = sorted(RessourceMonitor.cpus(), key=itemgetter(0))
        devices, loads = tuple(zip(*cpus))
        source = ColumnDataSource(self._bar_source(devices, ('load', loads)))

        val_range = Range1d(0, 100)
        name_range = FactorRange(factors=source.data['name'])
        kwargs['x_range'] = name_range
        kwargs['y_range'] = val_range
        kwargs['x_axis_label'] = 'device'
        kwargs['y_axis_label'] = '%'

        if not vertical:
            self._swapaxes(kwargs)

        p = figure(**kwargs, tools='hover', title='CPU load')

        if vertical:
            p.quad(left='name_low', right='name_high', top='load',
                   bottom='zeros', source=source)
        else:
            p.quad(left='zeros', right='load', top='name_high',
                   bottom='name_low', source=source)

        @self.change_consumer.register_subscriber
        def waiter():
            change = yield CPU
            while not self.terminated:
                loads = list(source.data['load'])
                loads[change.dev] = change.load
                source.data['load'] = loads
                change = yield

        return p

    def gpu_bars(self, vertical=True, **kwargs):
        gpus = sorted(RessourceMonitor.gpus_comb(), key=itemgetter(0))

        devices, free_mems, loads = tuple(zip(*gpus))
        max_free = max(free_mems) * 1.5
        source = ColumnDataSource(self._bar_source(devices,
                                                   ('free', list(free_mems)),
                                                   ('load', list(loads))))

        def waiter():
            change = yield GPUComb
            while not self.terminated:
                self._drop_in(source.data, 'load', change.dev, change.load)
                self._drop_in(source.data, 'free', change.dev, change.free)
                change = yield

        self.change_consumer.register_subscriber(waiter())

        name_range = FactorRange(factors=source.data['name'])

        def makefig(name_range, val_range, val_name, title, ylabel):
            kwargs['x_range'] = name_range
            kwargs['y_range'] = val_range
            kwargs['x_axis_label'] = 'device'
            kwargs['y_axis_label'] = ylabel

            if not vertical:
                self._swapaxes(kwargs)
            return figure(**kwargs, tools=[], title=title)

        p1, p2 = (makefig(name_range, Range1d(0, 100), 'load', 'GPU load', '%'),
                  makefig(name_range, Range1d(0, max_free), 'free',
                          'GPU free memory', 'MiB'))

        if vertical:
            p = vplot(p1, p2)
            p1.quad(left='name_low', right='name_high', top='load',
                    bottom='zeros', source=source)
            p2.quad(left='name_low', right='name_high', top='free',
                    bottom='zeros', source=source)
        else:
            p = hplot(p1, p2)
            p1.quad(left='zeros', right='load', top='name_high',
                    bottom='name_low', source=source)
            p2.quad(left='zeros', right='free', top='name_high',
                    bottom='name_low', source=source)
        p1.add_tools(HoverTool(tooltips=[('load', "@load")]))
        p2.add_tools(HoverTool(tooltips=[('free', "@free")]))
        return p

    def user_total_memusage(self, vertical=True, **kwargs):
        procs = sorted(RessourceMonitor.gpu_procs(), key=itemgetter(0))
        users = defaultdict(dict)
        for proc in procs:
            users[proc.owner][proc.pid] = proc.memusage

        def user2data(_users):
            owners = list(_users.keys())
            memusage = [sum(_users[owner]) for owner in owners]
            return self._bar_source(owners, ('memusage', memusage))

        source = ColumnDataSource(user2data(users))

        val_range = Range1d(0, max(source.data['memusage']) * 1.5)
        name_range = FactorRange(factors=source.data['name'])
        kwargs['x_range'] = name_range
        kwargs['y_range'] = val_range
        kwargs['x_axis_label'] = 'username'
        kwargs['y_axis_label'] = 'MiB'

        if not vertical:
            self._swapaxes(kwargs)

        p = figure(**kwargs, tools=[], title='Memory usage',
                   )

        if vertical:
            p.quad(left='name_low', right='name_high', top='memusage',
                   bottom='zeros', source=source)
        else:
            p.quad(left='zeros', right='memusage', top='name_high',
                   bottom='name_low', source=source)

        @self.change_consumer.register_subscriber
        def waiter():
            loop = asyncio.get_event_loop()
            change = yield GPUProcess
            print(change)
            while not self.terminated:
                users[change.owner][change.pid] = change.memusage

                # use "is" here to make sure it's the integer 0 which is a
                # sentinel for "dead"
                if change.memusage is 0:
                    print(change, 'marked as dead')
                    loop.call_later(10, users[change.owner].pop, change.pid)

                source.data.update(user2data(users))
                owners = list(users.keys())
                if name_range.factors != owners:
                    name_range.factors = owners
                change = yield

        return p

    def console(self, stdout=True, stderr=True, **kwargs):
        console_kwargs = dict(
            (key, kwargs[key]) for key in ['n', 'max_line_len', 'input_bottom']
            if key in kwargs)

        consoles = list()

        def waiter(sentinel, console: BokehConsole):
            change = yield sentinel
            if hasattr(sentinel, 'bytes'):
                    console.output_text(change.bytes.decode())
                    change = yield
            else:
                while not self.terminated:
                    console.output_text(change.text)
                    change = yield

        if stdout:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_subscriber(waiter(BytesStdOut, c))
            self.change_consumer.register_subscriber(waiter(TextStdOut, c))
            consoles.append(c)

        if stderr:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_subscriber(waiter(BytesStdErr, c))
            self.change_consumer.register_subscriber(waiter(TextStdErr, c))
            consoles.append(c)

        if len(consoles) == 1:
            return consoles[0].p
        return hplot(*(c.p for c in consoles))

    @staticmethod
    def _ellipsis_name(name):
        return '|'.join(re.findall('[a-z]+[^a-z]*\d+[^a-z]*', name))

    def progress(self, job_names, **kwargs):
        job_names = [self._ellipsis_name(nam) for nam in job_names]
        job_name2idx = dict((name, i) for i, name in enumerate(job_names))
        state2color = dict(zip(['dead', 'queued', 'init', 'running-1',
                                'running-2', 'running-3',
                                'complete'], reversed(RdYlGn7)))
        source = ColumnDataSource(self._bar_source(job_names, ('percent', 0),
                                                   ('color',
                                                    state2color['queued'])))
        p = figure(
            x_range=Range1d(0, 100), height=(100 + 30 * len(job_names)),
            width=650,
            y_range=FactorRange(factors=source.data['name']),
            tools=[HoverTool(tooltips=[("job", "@name"),
                                       ("percent", "@percent")])],
            x_axis_label='%', y_axis_label='jobname',
            title='Job progress'

        )

        p.quad(left='zeros', right='percent', top='name_high',
               bottom='name_low', fill_color='color', source=source)

        @self.change_consumer.register_subscriber
        def subscriber():
            change = yield JobProgress
            print(change)
            while not self.terminated:
                idx = job_name2idx.get(self._ellipsis_name(change.name), None)
                if idx is None:
                    warnings.warn('unknown job!: "{}"'.format(change))
                    continue
                self._drop_in(source.data, 'percent', idx, change.percent)

                if state2color[change.state] != source.data['color'][idx]:
                    self._drop_in(source.data, 'color', idx,
                                  state2color[change.state])
                change = yield

        return p

    def serve(self, host='localhost', port=5006, session_id='test'):
        url = 'http://' + host + ':' + str(port) + '/'
        curstate().reset()
        self.session = push_session(curdoc(),
                                    session_id=session_id,
                                    url=url)
        self.session.show()


def best_gpu():
    g = sorted(RessourceMonitor.gpus_mem(), key=attrgetter('free'),
               reverse=True)
    return g[0]


class ProgressMonitor:
    def __init__(self, job_name):
        self.job_name = job_name
        self.change_q = ChangeStream()
        self.gpus_wmean = self.GPU_WMeans()
        self.plot_gen = BokehPlots(self.change_q)
        self.ressource_mon = RessourceMonitor(self.change_q)
        self.figures = None
        self.terminated = False
        self._terminating = False

    class GPU_WMeans(UserDict):
        GPU_WMean = namedtuple('GPU_WMean', 'dev load free w_total')

        def __missing__(self, dev):
            return self.GPU_WMean(dev, 100, 0, 0)

        def __setitem__(self, dev, values):
            super().__setitem__(dev, self.GPU_WMean(dev, *values))

    def _update_gpu_wmean(self, load, free_mem, w_prev_total,
                          meas_hist: deque,
                          new_meas: GPUComb,
                          lookback_sec=60):
        """
        keep track of running gpu weighted mean (wmean)
        :param load: last calculated gpu load wmean
        :param free_mem:  last calculated free memory wmean
        :param w_total: total_weight for last calculated weighted means
        :param meas_hist: list of (timestamp, GPUProcess) pairs
        :param new_meas: GPUComb to use for updating
        :param lookback_sec: number of seconds to keep measuremts before discarding
        :return: (float, float, float)
        """
        now = time.time()
        if meas_hist:
            prev_t, prev_meas = meas_hist[-1]

            w_add = now - prev_t  # elapsed seconds between last measurement and now
            w_total = w_prev_total + w_add

            # add previous measurement to weighted means
            load = (load * w_prev_total + prev_meas.load * w_add) / w_total
            free_mem = (
                           free_mem * w_prev_total + prev_meas.free * w_add) / w_total
        else:
            load = new_meas.load
            free_mem = new_meas.free
            w_total = 0  # with only 1 measurement there is no "elapsed time"

        meas_hist.append((now, new_meas))

        # discard measurements older than 30 seconds. but do not discard down to
        # fewer than 2 measurements
        while meas_hist[0][0] + lookback_sec < now and len(meas_hist) > 2:
            discard_t, discard_meas = meas_hist.popleft()

            # elapsed seconds from discarded measurerment to next
            w_sub = meas_hist[0][0] - discard_t

            w_total, w_prev_total = (w_total - w_sub, w_total)

            # subtract discarded measurement from weighted means
            load = (load * w_prev_total - discard_meas.load * w_sub) / w_total
            free_mem = (
                           free_mem * w_prev_total - discard_meas.free * w_sub) / w_total

        return load, free_mem, w_total


    def gpu_running_average(self):
        @self.change_q.register_subscriber
        def subscriber():
            gpus_meas_hist = defaultdict(deque)
            gpus_wmeans = self.gpus_wmean
            new_meas = yield GPUComb
            while True:
                dev, load, free, w_total = gpus_wmeans[new_meas.dev]
                gpus_wmeans[dev] = self._update_gpu_wmean(load, free,
                                                          w_total,
                                                          gpus_meas_hist[dev],
                                                          new_meas)
                new_meas = yield

    async def best_gpu(self, min_free=2000, max_load=90,
                       blacklist=tuple(),
                       whitelist=tuple()) -> GPU_WMeans.GPU_WMean:
        """
        Find best available GPU that satisfy requirements
        "best" means lowest load
        :param min_free: minimum available free memory
        :param max_load: maximum load
        :param blacklist: list of devices that cannot be chosen
        :param whitelist: list of devices to choose from
            If empty, all devices not on exclude list can be chosen
        :return: GPU_WMean
        """
        best = self.GPU_WMeans.GPU_WMean(-1, 100.0, 0.0, 0)
        whitelist = set(whitelist)
        blacklist = set(blacklist)

        if blacklist and whitelist:
            whitelist = whitelist.difference(blacklist)

        if whitelist:
            def valid_dev(dev):
                return dev in whitelist

        elif blacklist:
            def valid_dev(dev):
                return dev not in blacklist
        else:
            def valid_dev(dev):
                return True

        def get_best(curr_best, gpu_wmean):
            # check if device is white/black listed
            if not valid_dev(gpu_wmean.dev):
                return curr_best

            # discard any GPU that does not satisfy load and memory requirements
            if gpu_wmean.free <= min_free or gpu_wmean.load >= max_load:
                return curr_best

            if gpu_wmean.load < best.load:
                return gpu_wmean
            return curr_best

        # check all known devices
        for gpu_wmean in self.gpus_wmean.values():
            best = get_best(best, gpu_wmean)

        # loop until a valid GPU has been found
        while best.dev < 0:
            new_meas = await self.change_q.wait_for_change(GPUComb)
            gpu_wmean = self.gpus_wmean[new_meas.dev]
            best = get_best(best, gpu_wmean)

        return best

    def layout(self, job_names):
        self.figures = vplot(hplot(self.plot_gen.progress(job_names),
                                   vplot(
                                       self.plot_gen.gpu_bars(vertical=True,
                                                              height=250),
                                       self.plot_gen.cpu_bars(vertical=False),

                                       self.plot_gen.user_total_memusage(
                                           height=400,
                                           width=400))),
                             self.plot_gen.console(n=50, max_line_len=80,
                                                   input_bottom=False))

    async def terminate(self):
        self._terminating = True
        await self.ressource_mon.terminate()
        await self.change_q.terminate()
        self.terminated = True
        self._terminating = False
        self.plot_gen.terminated = True
        self.plot_gen = None

    async def start(self, session_id, job_names):
        if self.terminated:
            warnings.warn('Trying to start a terminated {}'.format(self.__class__))
            return
        self.layout(job_names)
        self.gpu_running_average(),
        self.plot_gen.serve(port=5010, session_id=session_id)
        await asyncio.gather(self.ressource_mon.gpus_mon(),
                             self.ressource_mon.cpus_mon(),
                             self.change_q.start())

    def revive(self):
        self.terminated = False
        self.change_q.terminated = False
        self.ressource_mon.terminated = False
        if self.plot_gen is None:
            self.plot_gen = BokehPlots(self.change_q)


class BatchSemaphore(asyncio.locks.Semaphore):
    def __init__(self, n_locks, loop=None):
        super().__init__(n_locks, loop=loop)
        self.ids = deque(range(n_locks))

    def __await__(self):
        # To make "with await lock" work.
        yield from self.acquire()
        return IDContext(self.ids.popleft(), self)

    async def __aenter__(self):
        raise NotImplementedError()

    async def __aexit__(self, exc_type, exc, tb):
        raise NotImplementedError()


class IDContext:
    def __init__(self, context_id, semaphore: BatchSemaphore):
        self.id = context_id
        self.semaphore = semaphore

    async def __aenter__(self):
        return self.id

    def __enter__(self):
        return self.id

    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.ids.append(self.id)
        self.semaphore.release()

    def __exit__(self, exc_type, exc, tb):
        self.semaphore.ids.append(self.id)
        self.semaphore.release()
