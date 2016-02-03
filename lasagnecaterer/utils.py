# builtins
import _pyio
import inspect
import io
import _pyio  # for subclassing
import os
from collections import namedtuple, OrderedDict, deque, Generator, defaultdict, MutableSequence
from operator import attrgetter, itemgetter
from typing import Union, List
import re
import sys
import asyncio
import warnings
from decorator import contextmanager
from functools import singledispatch, partial
from subprocess import Popen, PIPE
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
from bokeh.models import ColumnDataSource, Range1d, FactorRange, HoverTool, TextEditor
from bokeh.plotting import figure, hplot, vplot, gridplot
from bokeh.client import push_session
from bokeh.io import curdoc
from bokeh.palettes import RdYlGn5
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
def any_to_stream(inp: Union[str, ioabc.InputStream, bytes],
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


@any_to_stream.register(str)
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


@any_to_stream.register(ioabc.InputStream)
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
        return any_to_stream(inp.read())

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


@any_to_stream.register(bytes)
def bytes_to_stream(inp: bytes, output_type: Union[str, bytes] = str,
                    force_seekable=False):
    if output_type is bytes:
        return io.BytesIO(inp)
    return io.StringIO(inp.decode())


from elymetaclasses.events import ChainedPropsMetaClass
from abc import ABCMeta


class ChainPropsABCMetaclass(ChainedPropsMetaClass, ABCMeta):
    pass


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
        self._rotate = 1 if input_bottom else -1
        self._pos = 0 if input_bottom else -1
        super(BokehConsole, self).__init__()

    def make_source(self):
        return ColumnDataSource({'text': [''] * self.n,
                                 'zeros': [0] * self.n,
                                 'line': list(range(self.n))})

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

    def output_text(self, s):
        for line in s.split('\n'):
            if len(line) <= self.max_line_len:
                self._push_line(line)
            else:
                tokens = list()
                i = -1
                for token in line.split(' '):
                    i += 1 + len(token)
                    if i > self.max_line_len:
                        self._push_line(' '.join(tokens))
                        tokens = [token]
                        i = len(token)
                    else:
                        tokens.append(token)


JobProgress = namedtuple('JobProgress', 'name percent state')

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
    def __init__(self, change_stream: asyncio.Queue,
                 async_exec=async_subprocess,
                 normal_exec=Popen):
        # self.sources = self.init_sources()
        self.async_exec = async_exec
        self.normal_exec = normal_exec
        self.change_stream = change_stream
        self.terminated = False

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

    async def gpus_mon(self, loop: asyncio.BaseEventLoop = None,
                       ignore=tuple()):
        subproc_exec = self.async_exec
        nv2cuda, pid2owner = await asyncio.gather(
            self.nv2cuda_coro(subproc_exec),
            self.pid2owner_coro(subproc_exec))

        p = await self.async_exec('nvidia-smi', '-l', '1',
                                  stdout=asyncio.subprocess.PIPE,
                                  stderr=FNULL)

        loop = loop or asyncio.get_event_loop()
        gpus = dict()
        gpu_nvprocs = dict()
        do_GPUComb = GPUComb not in ignore
        do_GPUProcess = GPUProcess not in ignore
        nvdev = None
        tasks = list()
        seen_pids = list()
        while not self.terminated:
            line = await p.stdout.readline()
            line = line.decode()
            if do_GPUComb:
                nvdev = nv_line2nvdev(line, nvdev)
                nvgpu = nv_line2GPUNv(line, nvdev)

                # a gpu was found in stdout
                if nvgpu:
                    prev_gpu = gpus.get(nvgpu.nvdev, None)
                    # has anything changed?
                    if prev_gpu != nvgpu[1:]:
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
                                                            pid2owner[dead_proc.pid],
                                                            nv2cuda[dead_proc.nvdev],
                                                            0))
                    gpu_nvprocs.pop(dead_proc.pid)
                seen_pids.clear()


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

    async def cpus_mon(self):
        subproc_exec = self.async_exec
        p = await subproc_exec('top', '-b', '-p0',
                               stdout=PIPE)
        cpus = dict()
        while not self.terminated:
            line = (await p.stdout.readline()).decode()
            cpu = lines2CPUs(line)
            if cpu:
                cpu = cpu[0]
                prev_load = cpus.get(cpu.dev, None)
                if prev_load != cpu.load:
                    cpus[cpu.dev] = cpu.load
                    await self.change_stream.put(cpu)

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


BytesStdOut = namedtuple('BytesStdOut', 'bytes')
TextStdOut = namedtuple('TextStdOut', 'text')
BytesStdErr = namedtuple('BytesStdErr', 'bytes')
TextStdErr = namedtuple('TextStdErr', 'text')


class ChangeStream(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentinel2waiters = defaultdict(list)
        self.terminated = False

    class BytesRedirecter:
        def __init__(self, change_stream: asyncio.Queue, copy_out, wrap_class=BytesStdOut):
            self.change_stream = change_stream
            self._copy_out = copy_out
            self.wrap_class = wrap_class
            self.put_bytes = True if hasattr(wrap_class, 'bytes') else False

        def write(self, b):
            is_bytes = isinstance(b, bytes)
            # no change needed
            if (is_bytes and self.put_bytes) or (not is_bytes and not self.put_bytes):
                    self.change_stream.put_nowait(self.wrap_class(b))

            # is bytes, but should be str
            elif is_bytes:
                self.change_stream.put_nowait(self.wrap_class(b.decode()))

            # is not bytes, but should be
            else:
                self.change_stream.put_nowait(self.wrap_class(b.encode()))

            if self._copy_out:
                self._copy_out.write(b)

    class TextRedirecter:
        def __init__(self, change_stream:  asyncio.Queue, copy_out, wrap_class=TextStdOut):
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

    def register_waiter(self, waiter_coro):
        if inspect.isfunction(waiter_coro):
            waiter_coro = waiter_coro()
        sentinel = next(waiter_coro)
        self.sentinel2waiters[sentinel].append(waiter_coro)

    async def start(self):
        while not self.terminated:
            change = await self.get()
            for sentinel, waiters in self.sentinel2waiters.items():
                if isinstance(change, sentinel):
                    dead_waiters = list()
                    for waiter in waiters:
                        try:
                            waiter.send(change)
                        except StopIteration:
                            dead_waiters.append(waiter)
                    for waiter in dead_waiters:
                        waiters.remove(waiter)

@contextmanager
def redirect_to_changestream(change_stream: asyncio.Queue, err=False):
    stdout = sys.stdout



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
        kwargs['x_axis_label'], kwargs['y_axis_label'] = kwargs['y_axis_label'], kwargs[
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

        @self.change_consumer.register_waiter
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

        self.change_consumer.register_waiter(waiter())

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
                  makefig(name_range, Range1d(0, max_free), 'free', 'GPU free memory', 'MiB'))


        if vertical:
            p = vplot(p1,p2)
            p1.quad(left='name_low', right='name_high', top='load',
                   bottom='zeros', source=source)
            p2.quad(left='name_low', right='name_high', top='free',
                   bottom='zeros', source=source)
        else:
            p = hplot(p1,p2)
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

        val_range = Range1d(0, max(source.data['memusage'])*1.5)
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


        @self.change_consumer.register_waiter
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
        console_kwargs = dict((key, kwargs[key]) for key in ['n', 'max_line_len', 'input_bottom'] if key in kwargs)

        consoles = list()

        def waiter(sentinel, console: BokehConsole):
            change = yield sentinel
            if hasattr(sentinel, 'bytes'):
                while not self.terminated:
                    console.output_text(change.bytes.decode())
                    change = yield
            else:
                while not self.terminated:
                    console.output_text(change.text)
                    change = yield

        if stdout:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_waiter(waiter(BytesStdOut, c))
            self.change_consumer.register_waiter(waiter(TextStdOut, c))
            consoles.append(c)

        if stderr:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_waiter(waiter(BytesStdErr, c))
            self.change_consumer.register_waiter(waiter(TextStdErr, c))
            consoles.append(c)

        if len(consoles) == 1:
            return consoles[0].p
        return hplot(*(c.p for c in consoles))

    def progress(self, job_names, **kwargs):
        job_name2idx = dict((name, i) for i, name in enumerate(job_names))
        state2color = dict(zip(['dead', 'queued', 'init', 'running',
                                 'complete'], reversed(RdYlGn5)))
        source = ColumnDataSource(self._bar_source(job_names, ('percent', 0),
                                                   ('color', RdYlGn5[1])))
        p = figure(
            x_range=Range1d(0, 100), height=(100 + 30 * len(job_names)),
            width=400,
            y_range=FactorRange(factors=source.data['name']),
            tools=[HoverTool(tooltips=[("job", "@name"),
                                       ("percent", "@percent")])],
            x_axis_label='%', y_axis_label='jobname',
            title='Job progress'

        )

        p.quad(left='zeros', right='percent', top='name_high',
                  bottom='name_low', fill_color='color', source=source)

        @self.change_consumer.register_waiter
        def waiter():
            change = yield JobProgress
            while not self.terminated:
                idx = job_name2idx.get(change.name, None)
                if None:
                    warnings.warn('unknown job!: "{}"'.format(change))
                self._drop_in(source.data, 'percent', idx, change.percent)

                if state2color[change.state] != source.data['color'][idx]:
                    self._drop_in(source.data, 'color', idx, state2color[change.state])
                change = yield
        return p

    def serve(self, host='localhost', port=5006, session_id='test'):
        url = 'http://' + host + ':' + str(port) + '/'
        self.session = push_session(curdoc(),
                                    session_id=session_id,
                                    url=url)
        self.session.show()


def best_gpu():
    g = sorted(RessourceMonitor.gpus_mem(), key=attrgetter('free'),
               reverse=True)
    return g[0]


class ProgressUpdater:
    def __init__(self, job_name):
        self.job_name = job_name
        self.source = None
        self._waiter = None
        self.session = None
        self.changes = OrderedDict()
        self.console_stream = AsyncBytesIO()
        self.terminated = False
        self.progress = OrderedDict()
        self.loop = None
        self.ressources = None
        self.console = None

    async def terminate(self):
        self.terminated = True
        await self.wakeup_waiter()

    def init_data_source(self, progress: OrderedDict):
        source = ColumnDataSource({
            'progress': list(progress.values()),
            'zeros': [0] * len(progress),
            'prefix': list(progress.keys()),
            'prefix_top': [p + ':0.9' for p in progress.keys()],
            'prefix_bottom': [p + ':0.1' for p in progress.keys()]
        })
        return source

    def make_plot(self):
        source = self.source
        prog = figure(
            x_range=Range1d(0, 100), height=700, width=400,
            y_range=FactorRange(factors=source.data['prefix']),
            tools=[HoverTool(tooltips=[("job", "@prefix"),
                                       ("progress", "@progress")])]
        )
        prog.quad(left='zeros', right='progress', top='prefix_top',
                  bottom='prefix_bottom', source=source)

        res = [figure(y_range=Range1d(0, max(r.data['progress'])),
                      x_range=FactorRange(factors=r.data['prefix']),
                      height=300, width=800,
                      tools=[HoverTool(tooltips=[("dev", "@prefix"),
                                                 ("value", "@progress")])]) for
               r in self.ressources]

        for fig, r in zip(res, self.ressources):
            fig.quad(left='prefix_bottom', right='prefix_top',
                     top='progress', bottom='zeros', source=r)

        self.console = BokehConsole(n=40, max_line_len=200, input_bottom=False)

        """
        res = [charts.Bar(self.ressources[1].to_df(), label='gpu_dev',
                               values='gpu_free', height=100, width=200),
                    charts.Bar(self.ressources[0].to_df(), label='cpu_dev',
                               values='cpu_load', width=600)
                    ]
        """

        vplot(hplot(prog, vplot(*res)), self.console.p)
        # gridplot([[prog, res[0]], [res[1]]])
        # curdoc().add_root(hplot(prog, res))

    def push(self):
        # open a session to keep our local document in sync with server
        self.session = push_session(curdoc(),
                                    session_id=self.job_name,
                                    url='http://localhost:5010/')
        self.session.show()

    async def wakeup_waiter(self):
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            if not waiter.cancelled():
                waiter.set_result(None)

    async def wait_for_data(self):
        if self._waiter is not None:
            raise RuntimeError('%s() called while another coroutine is '
                               'already waiting for incoming data')

        self._waiter = asyncio.futures.Future(loop=self.loop)
        try:
            await self._waiter
        finally:
            self._waiter = None

    async def progress_updater(self):
        while True:
            await self.wait_for_data()
            if not self.changes:
                continue

            self.progress.update(self.changes)
            self.changes.clear()
            self.source.data['progress'] = list(self.progress.values())

    async def ressource_updater(self):
        while True:
            self.ressources[0].data['progress'] = [g.free for g in
                                                   await gpus_coro()]
            self.ressources[1].data['progress'] = [c.load for c in
                                                   await cpus_coro()]
            await asyncio.sleep(2.5)

    async def console_updater(self):
        console = self.console
        assert isinstance(console, BokehConsole)
        while True:
            data = await self.console_stream.readline()
            console.output_text(data.decode())

    async def start(self, progress, loop):
        self.loop = loop
        self.source = self.init_data_source(progress)
        gpu = OrderedDict(('GPU {0}'.format(g.dev), g.free) for g in gpus())
        cpu = OrderedDict(('CPU {0}'.format(g.dev), g.load) for g in cpus())
        self.ressources = (
            self.init_data_source(gpu), self.init_data_source(cpu))
        # self.init_ressource_source()
        self.progress = progress
        self.make_plot()
        self.push()
        await asyncio.gather(self.progress_updater(), self.ressource_updater(),
                             self.console_updater())
