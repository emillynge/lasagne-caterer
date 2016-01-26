"""
For orchestrating the lasagna making process
"""
# builtins
import os
import tempfile
from asyncio.subprocess import PIPE
from collections import (namedtuple, OrderedDict, defaultdict, deque)
from typing import List

import asyncio
import numpy as np


# github packages
import theano
import tqdm

from elymetaclasses.abc import io as ioabc

# relative
from .utils import any_to_stream, pbar, best_gpu
from .recipe import LasagneBase
from .oven import BufferedBatchGenerator, CharmappedBatchGenerator, FullArrayBatchGenerator
from .fridge import ClassSaveLoadMixin, TupperWare
from .menu import Options

class BaseCook(ClassSaveLoadMixin):
    def __init__(self, opt: Options,
                 oven: FullArrayBatchGenerator,
                 recipe: LasagneBase,
                 fridge):

        self.opt = opt
        self.oven = oven
        self.recipe = recipe
        self.fridge = fridge
        box = fridge.shelves['cook']
        assert isinstance(box, TupperWare)
        self.box = box
        self.open_shop()

    def open_shop(self):
        """
        Actions to be taken after everything has been taken out of the fridge
        :return:
        """
        if 'all_params' in self.fridge.shelves['recipe']:
            self.recipe.saved_params = self.fridge.shelves['recipe'].all_params

    def close_shop(self):
        """
        Actions to be taken before we put everything in the fridge
        :return:
        """
        self.fridge.shelves['recipe']['all_params'] = self.recipe.all_params

    def to_dict(self):
        self.close_shop()
        return super().to_dict()


class CharmapCook(BaseCook):
    def open_shop(self):
        super().open_shop()
        oven = self.oven
        assert isinstance(oven, CharmappedBatchGenerator)
        self.oven = oven
        if 'charmap' in self.fridge.shelves['oven']:
            self.oven.charmap = self.fridge.shelves['oven'].charmap

    def close_shop(self):
        super().close_shop()
        self.fridge.shelves['oven']['charmap'] = self.oven.charmap


class LasagneTrainer(CharmapCook):
    def open_shop(self):
        super().open_shop()
        oven = self.oven
        assert isinstance(oven, FullArrayBatchGenerator)
        self.oven = oven

    def train(self, epochs) -> List[None]:
        for x, y in self.oven.iter_epoch(epochs):
            yield self.recipe.f_train_noreturn(x, y)

    def train_err(self, batches):
        for x, y in self.oven.iter_batch(batches):
            yield self.recipe.f_cost(x, y)

    def test(self, epochs):
        for x, y in self.oven.iter_epoch(epochs, part='test'):
            yield self.recipe.f_cost(x, y)

    def auto_train(self):
        required_opt = ('start_epochs', 'decay_epochs')
        if any(o not in self.opt for o in required_opt):
            raise ValueError('Options missing. Need {}'.format(required_opt))

        s_ep = self.opt.start_epochs
        print('Start train {} epochs\n'.format(s_ep))

        pb, msg = pbar('batches', s_ep * self.oven.batches_per_epoch.train, '')
        pb.start()
        for _ in pb(self.train(s_ep)):
            pass

        train_err_batches = self.oven.batches_per_epoch.train // 4
        test_err_hist = list(self.test(1))
        train_err_hist = list(self.train_err(train_err_batches))
        l1 = len(train_err_hist)
        l2 = len(test_err_hist)
        te_err = np.mean(test_err_hist[-l2:])
        tr_err = np.mean(train_err_hist[-l1:])

        message = lambda te, tr: 'tr: {0:1.2f} te: {1:1.2f}'.format(te, tr)
        MEM = 5
        params = deque([np.copy(self.recipe.all_params)])    #BytesIO() for _ in range(MEM))
        test_err = deque([np.mean(test_err_hist)])  #10.0**10 for _ in range(MEM))

        pb, msg = pbar('epochs', self.opt.decay_epochs, message(te_err, tr_err))
        for _i in pb(range(self.opt.decay_epochs)):
            i = _i + 1
            try:
                for _ in self.train(1):
                    pass

                train_err_hist.extend(self.train_err(train_err_batches))
                test_err_hist.extend(self.test(1))
                te_err = np.mean(test_err_hist[-l2:])
                tr_err = np.mean(train_err_hist[-l1:])
                message(te_err, tr_err)

            except KeyboardInterrupt:
                break

            # build or rotate deque
            if i < MEM:
                params.appendleft(np.copy(self.recipe.all_params))
                test_err.appendleft(te_err)
            else:
                params.rotate()
                test_err.rotate()

            min_test_err = np.min(test_err)
            # store or discard
            if min_test_err >= te_err: # current error is as good or better than all previous
                # store
                params[0] = np.copy(self.recipe.all_params)
                test_err[0] = te_err

            elif min_test_err == test_err[0]: # oldest model is best. stop
                # give up
                print('Early stopping..', test_err)
                break
            else:
                # discard
                pass
        else:
            print('Did not find a minimum. Stopping')

        idx_best = np.argmax(test_err)
        print('Stopped training. choosing {}'.format(idx_best), test_err)
        self.recipe.set_all_params(params[idx_best])
        self.box['test_error_hist'] = test_err_hist
        self.box['train_error_hist'] = train_err_hist


class AsyncHeadChef(LasagneTrainer):
    compiled_base_dir = theano.config.base_compiledir

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = asyncio.get_event_loop()
        self.startup_lock = asyncio.locks.Lock(loop=self.loop)
        self.semaphore = None
        self.basemodel_path = None
        self.processes = list()

    def make_feature_net(self, **features):
        feature_name, feature_list = features.popitem()
        for feature in feature_list:
            res = [(feature_name, feature)]
            if not features:
                yield res
                continue

            for iter_res in self.make_feature_net(**features):
                yield res + iter_res

    def featurenet_train(self, out_dir, concurrent=5, **features):
        try:
            os.mkdir(out_dir)
        except IOError:
            pass

        self.fridge.save(out_dir + os.sep + 'basemodel.lfr' )
        fd_bm, self.basemodel_path = tempfile.mkstemp(suffix='.lrf', prefix='basemodel',
                                              dir=os.path.abspath('.'))
        try:
            self.fridge.save(fd_bm)
        finally:
            fd_bm.close()

        total = sum(len(val) for val in features.values())
        self.semaphore = asyncio.Semaphore(concurrent)
        to_do = (self.train_model(overrides, out_dir) for overrides
                                 in self.make_feature_net(**dict(features)))
        self.loop.run_until_complete(self.trainer_coro(to_do, total))
        self.loop.close()

    async def trainer_coro(self, to_do, total):
        to_do_iter = asyncio.as_completed(to_do)
        to_do_iter = tqdm.tqdm(to_do_iter, total=total)

        for future in to_do_iter:
            await future

    async def train_model(self, overrides, out_dir):
        prefix = '_'.join('-'.join(override for override in overrides))
        fname = prefix + '.lfr'

        with await self.semaphore:
            # get a compiledir
            comp_dir = self.compiled_base_dir + '/semaphore-1-' + str(self.semaphore._value)

            with open(prefix + '.log', 'wb', buffering=1) as logfile:
                with await self.startup_lock:
                    # find best gpu. Wait if no suitable gpu exists
                    gpu = best_gpu()
                    while gpu.free < 2000:
                        print('Best gpu: {} - Waiting...'.format(gpu))
                        await asyncio.sleep(1)
                        gpu = best_gpu()

                    # define environment
                    _env = dict(os.environ)
                    _env['THEANO_FLAGS'] = 'base_compiledir={0},device=gpu{1}'.format(comp_dir, gpu.dev)

                    # make sure compiledir exists
                    try:
                        os.mkdir(comp_dir)
                    except FileExistsError:
                        pass

                    # Start up a worker
                    print('Started on ' + prefix + ' using ' + _env['THEANO_FLAGS'])

                    p = await asyncio.create_subprocess_exec('mypython3', '-i',
                                                             self.basemodel_path,
                                                             stdout=logfile,
                                                             stderr=logfile,
                                                             stdin=PIPE,
                                                             env=_env)
                    assert isinstance(p, asyncio.subprocess.Process)
                    self.processes.append(p)
                    stdin = p.stdin
                    assert isinstance(stdin, asyncio.streams.StreamWriter)

                    # send commands to worker to change features
                    for feature_name, value in overrides:
                        stdin.write('obj.opt.{0} = {1}\n'.format(feature_name, value))
                    await stdin.drain()

                    # startup the training
                    stdin.write('obj.auto_train()\nobj.save({0})'.format(fname))
                    await stdin.drain()

                    # close stream so process will terminate
                    stdin.close()

                    # wait some time for the worker to actually start using the
                    # GPU before releasing startup lock
                    await asyncio.sleep(20)  # wait some time to release lock
                # startup lock released here

                await p.wait()
                self.processes.remove(p)
            print('Completed ' + fname)

"""
import asyncio
import tqdm

from time import sleep
import signal
loop = asyncio.get_event_loop()
import theano
import os
from asyncio.subprocess import PIPE
import re

env = os.environ
#FNULL = open(os.devnull, 'w')
import shutil
def remove_lock(cmp_dir):
    for root, dirs, files in os.walk(cmp_dir):
        if 'lock_dir' in dirs:
            shutil.rmtree(root + '/lock_dir')
        #if 'lock' in files and 'lock_dir' in root:
        #    for f in files:
        #        if f == 'lock':
        #
        #            os.remove(root + '/' + f)

processes = list()

pbar_regx = re.compile(b'(\d+%) \|#+')
start_up_lock = asyncio.locks.Lock(loop=loop)
async def train_model(dropout, semaphore):
    with (await semaphore):
        fname = 'models/dropoutcv3/n3l512do_{:3}'.format(int(dropout*100)).replace(' ', '0')
        cmp_dir = compiled_base_dir + '/semaphore-1-' + str(semaphore._value)
        with open(fname + '.log', 'wb', buffering=1) as logfile:

            with await start_up_lock:
                gpu = best_gpu()
                while gpu.free < 2000:
                    print('Best gpu: {} - Waiting...'.format(gpu))
                    await asyncio.sleep(1)
                    gpu = best_gpu()

                _env = dict(env)
                _env['THEANO_FLAGS'] = 'base_compiledir={0},device=gpu{1}'.format(cmp_dir, gpu.dev)
                try:
                    os.mkdir(cmp_dir)
                except FileExistsError:
                    pass

                print('Started on ' + fname + ' using ' + _env['THEANO_FLAGS'])

                #sleep(10)

                p = await asyncio.create_subprocess_exec('mypython3', u'basemodel.lm',
                                                         '--train', fname + '.lm',
                                                         '--dropout', str(dropout),
                                                         stdout=PIPE,
                                                         stderr=PIPE,
                                                         env=_env)
                await asyncio.sleep(10)  # wait some time to release lock

            def read_to_log():
                #print(p.stdout._buffer)
                if p.stderr._buffer:
                    f = pbar_regx.findall(p.stderr._buffer)
                    if f:
                        print('{0} at {1}'.format(fname, f[0].decode()))
                logfile.write(p.stdout._buffer)
                p.stdout._buffer.clear()
                logfile.write(p.stderr._buffer)
                p.stderr._buffer.clear()
                #print('flushing')
                logfile.flush()

            processes.append(p)
            while True:
                try:
                    #print('waiting')
                    await asyncio.wait_for(p.wait(), 1)
                    read_to_log()
                    break
                except asyncio.TimeoutError:
                    #print('timed out')
                    read_to_log()
                    #logfile.flush()
                    #pass

            processes.remove(p)
            remove_lock(cmp_dir)
        print('Completed ' + fname)



        #counter[status] += 1  # <12>

    #return counter  # <13>



dropouts = np.arange(0.0, .95, .05) #[.1, .2, .3, .4, .5, .6, .7, .8]
try:
    loop.run_until_complete(trainer_coro(dropouts, 5))
    loop.close()
finally:
    for p in processes:
        try:
            print('interrupting {}'.format(p))
            p.send_signal(signal.SIGINT)
            sleep(5)
        except Exception:
            pass
        try:
            print('terminating {}'.format(p))
            p.terminate()
            sleep(1)
        except Exception:
            pass

        try:
            print('killing {}'.format(p))
            p.kill()
        except Exception:
            pass
"""