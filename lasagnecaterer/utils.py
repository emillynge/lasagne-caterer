# builtins
import _pyio
import io
import os
from collections import namedtuple
from operator import attrgetter
from typing import Union, re

import sys
from decorator import contextmanager
from functools import singledispatch

# pip
from progressbar import (ProgressBar, Percentage, SimpleProgress, Bar, Widget)

# github
from subprocess import Popen, PIPE

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
                  output_type: Union[str, bytes]=str,
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
def str_to_stream(inp: str, output_type: Union[str, bytes]=str,
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
def stream_to_stream(inp: ioabc.InputStream, output_type: Union[str, bytes]=str,
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
def bytes_to_stream(inp: bytes, output_type: Union[str, bytes]=str,
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
                       widgets=[SimpleProgress(),
                                what, msg,
                                Percentage(), Bar(), '\n']), msg


GPU = namedtuple('GPU', 'dev free')
dev_used_tot = re.compile('Device  (\d).+ (\d+) of (\d+) MiB Used')

def gpus():
    for line in Popen(['cuda-smi'], stdout=PIPE).stdout.read().decode('utf8').split('\n')[:-1]:
        dev, used, tot = dev_used_tot.findall(line)[0]
        free = int(tot) - int(used)
        yield GPU(int(dev), free)

def best_gpu():
    g = sorted(gpus(), key=attrgetter('free'), reverse=True)
    return g[0]
