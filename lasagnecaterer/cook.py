"""
For orchestrating the lasagna making process
"""
# builtins
import os
import tempfile
import warnings
from asyncio.subprocess import PIPE
from collections import (namedtuple, OrderedDict, defaultdict, deque)
from functools import partial
from typing import List
import re
import asyncio
from aiohttp import web
import zipfile
import atexit

# pip
import numpy as np
import tqdm
import theano

# github packages
from elymetaclasses.abc import io as ioabc

# relative
from .utils import any_to_stream, pbar, best_gpu, ProgressMonitor, ChangeStream, \
    JobProgress, async_subprocess, BatchSemaphore
from .recipe import LasagneBase
from .oven import BufferedBatchGenerator, CharmappedBatchGenerator, \
    FullArrayBatchGenerator
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
        self.fridge.shelves['recipe'][
            'all_params'] = self.recipe.get_all_params_copy()

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
        for x, y in self.oven.iter_batch(batches, part='train'):
            yield self.recipe.f_cost(x, y)

    def val(self, epochs):
        for x, y in self.oven.iter_epoch(epochs, part='val'):
            yield self.recipe.f_cost(x, y)

    def test(self, epochs):
        for x, y in self.oven.iter_epoch(epochs, part='test'):
            yield self.recipe.f_cost(x, y)

    def auto_train(self, pbar=pbar):
        required_opt = ('start_epochs', 'decay_epochs')
        print(self.val(1))
        if any(o not in self.opt for o in required_opt):
            raise ValueError('Options missing. Need {}'.format(required_opt))

        s_ep = self.opt.start_epochs
        print('Start train {} epochs\n'.format(s_ep))

        train_err_batches = self.oven.batches_per_epoch.train
        val_err_hist = list(self.val(1))
        train_err_hist = list(self.train_err(train_err_batches))
        l1 = len(train_err_hist)
        l2 = len(val_err_hist)
        te_err = np.mean(val_err_hist[-l2:])
        tr_err = np.mean(train_err_hist[-l1:])
        message = lambda te, tr: 'te: {0:1.3f} tr: {1:1.3f}'.format(te, tr)
        pb, msg = pbar('batches', s_ep * train_err_batches,
                       message(te_err, tr_err))
        pb.start()

        for j in range(self.opt.start_epochs):
            for _ in self.train(1):
                pb.update(pb.currval + 1)

            train_err_hist.extend(self.train_err(train_err_batches))
            val_err_hist.extend(self.val(1))
            te_err = np.mean(val_err_hist[-l2:])
            tr_err = np.mean(train_err_hist[-l1:])
            msg.message = message(te_err, tr_err)
        pb.finish()
        # // 4
        # val_err_hist = list(self.val(1))
        # train_err_hist = list(self.train_err(train_err_batches))

        MEM = 10
        params = deque([
            self.recipe.get_all_params_copy()])  # BytesIO() for _ in range(MEM))
        val_err = deque(
            [np.mean(val_err_hist)])  # 10.0**10 for _ in range(MEM))

        pb, msg = pbar('epochs', self.opt.decay_epochs, message(te_err, tr_err))
        for _i in pb(range(self.opt.decay_epochs)):
            i = _i + 1
            try:
                for _ in self.train(1):
                    # print(_)
                    pass

                train_err_hist.extend(self.train_err(train_err_batches))
                val_err_hist.extend(self.val(1))
                te_err = np.mean(val_err_hist[-l2:])
                tr_err = np.mean(train_err_hist[-l1:])
                msg.message = message(te_err, tr_err)

            except KeyboardInterrupt:
                break

            # build or rotate deque
            if i < MEM:
                params.appendleft(self.recipe.get_all_params_copy())
                val_err.appendleft(te_err)
            else:
                params.rotate()
                val_err.rotate()

            min_test_err = np.min(val_err)
            # store or discard
            if min_test_err >= te_err:  # current error is as good or better than all previous
                # store
                params[0] = self.recipe.get_all_params_copy()
                val_err[0] = te_err

            elif min_test_err == val_err[0]:  # oldest model is best. stop
                # give up
                print('Early stopping..', val_err)
                break
            else:
                # discard
                pass
        else:
            print('Did not find a minimum. Stopping')

        pb.finish()
        idx_best = np.argmin(val_err)
        print('Stopped training. choosing model #{} -> '.format(idx_best),
              val_err)
        self.recipe.set_all_params(params[idx_best])
        self.box['params'] = params[idx_best]
        self.box['val_error_hist'] = val_err_hist
        self.box['train_error_hist'] = train_err_hist


class LearningRateMixin:
    def __init__(self, *args, **kwargs):
        self._epochs = 0
        super().__init__(*args, **kwargs)

    def train(self, epochs) -> List[None]:
        step = max(self._epochs - self.recipe.opt.start_epochs, 0)

        for x, y in self.oven.iter_epoch(epochs, part='train'):
            yield self.recipe.f_train_noreturn(step=step)(x, y)
        self._epochs += epochs

    def train_err(self, batches):
        for x, y in self.oven.iter_batch(batches, part='train'):
            yield self.recipe.f_cost(x, y)

    def val(self, epochs):
        for x, y in self.oven.iter_epoch(epochs, part='val'):
            yield self.recipe.f_cost(x, y)

    def test(self, epochs):
        for x, y in self.oven.iter_epoch(epochs, part='test'):
            yield self.recipe.f_cost(x, y)


class AsyncHeadChef(LasagneTrainer):
    compiled_base_dir = theano.config.base_compiledir

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = asyncio.get_event_loop()
        self.startup_lock = asyncio.locks.Lock(loop=self.loop)
        self.semaphore = None
        self.basemodel_path = None
        self.progress_mon = ProgressMonitor('test')
        self.active_procs = list()

    def make_feature_net(self, **features):
        feature_name = sorted(features.keys())  # sort alphabetically
        feature_name = sorted(feature_name,  # sort number of possible values
                              key=lambda k: len(features[k]))[0]
        feature_list = features.pop(feature_name)
        for feature in feature_list:
            res = [(feature_name, feature)]
            if not features:
                yield res
                continue

            for iter_res in self.make_feature_net(**features):
                yield res + iter_res

    def featurenet_train(self, out_dir, session_id, concurrent=3, **features):
        try:
            os.mkdir(out_dir)
        except IOError:
            pass

        self.fridge.save(out_dir + os.sep + 'basemodel.lfr')
        fd_bm, self.basemodel_path = tempfile.mkstemp(suffix='.lrf',
                                                      prefix='basemodel',
                                                      dir=os.path.abspath('.'))
        with open(fd_bm, 'wb') as fp:
            self.fridge.save(fp)
        atexit.register(partial(os.remove, self.basemodel_path))

        self.semaphore = BatchSemaphore(concurrent)
        override_combinations = list(self.make_feature_net(**dict(features)))
        to_do = [self.train_model(overrides, out_dir) for overrides
                 in override_combinations]

        job_names = [self.prefix_from_overrides(o) for o in
                     override_combinations]

        try:
            with self.progress_mon.change_q.redirect_stdout(copy=True):
                with self.progress_mon.change_q.redirect_stderr(copy=True):
                    coros = asyncio.gather(self.progress_mon.start(session_id,
                                                                   job_names),
                                           self.trainer_coro(to_do))
                    self.loop.run_until_complete(coros)
                    self.loop.close()
        finally:
            self.terminate()

    def terminate(self):
        while self.active_procs:
            proc = self.active_procs.pop()
            assert isinstance(proc, asyncio.subprocess.Process)
            try:
                proc.terminate()
                proc.kill()
            except:
                pass

        if not self.progress_mon.terminated:
            self.progress_mon.terminate()

    async def trainer_coro(self, to_do):
        n = len(to_do)
        to_do_iter = asyncio.as_completed(to_do)
        to_do_iter = tqdm.tqdm(to_do_iter, total=n)
        i = 0
        for future in to_do_iter:
            await future
            i += 1
        self.progress_mon.terminate()

    async def write_to_log(self, stream: asyncio.streams.StreamReader,
                           logfile: ioabc.OutputStream,
                           prefix):
        pbar_regx = re.compile(b'(\d+)%\ ?\|#+')
        state = 1
        while True:
            # print('waiting on ' + prefix)
            data = await stream.readline()  # this should be bytes
            # print('read {} bytes'.format(len(data)))
            if not data and stream.at_eof():
                # print('breaking')
                break

            for percent in [int(percent) for percent in
                            pbar_regx.findall(data)]:
                self.job_progress(prefix, percent,
                                  'running-{}'.format(state))
                if percent == 100:
                    state += 1
            logfile.write(data.decode())

    def job_progress(self, prefix, percent, stage):
        self.progress_mon.change_q.put_nowait(
            JobProgress(prefix, percent, stage))

    @staticmethod
    def prefix_from_overrides(overrides):
        return '_'.join(
            '{0}-{1:02.0f}'.format(key, val * 100 if val <= 1 and isinstance(val,
                                                                            float) else val)
            for key, val in overrides)

    async def train_model(self, overrides, out_dir, overwrite=False):
        prefix = self.prefix_from_overrides(overrides)
        fname = out_dir + os.sep + prefix + '.lfr'
        if os.path.isfile(fname) and not overwrite:
            warnings.warn('{} already exist'.format(fname))
            try:
                zipfile.ZipFile(fname)
            except zipfile.BadZipFile:
                warnings.warn('not a valid zip file -> overwrite'.format(fname))
            else:
                warnings.warn('Skipping...')
                self.job_progress(prefix, 100, 'complete')
                return

        with (await self.semaphore) as sem_id:
            self.job_progress(prefix, 33, 'init')
            # get a compiledir
            comp_dir = self.compiled_base_dir + '/semaphore-1-' + str(sem_id)

            # make sure we only start 1 new process at a time so we don't put
            # all on same GPU
            with (await self.startup_lock):
                self.job_progress(prefix, 66, 'init')

                # find best gpu. Wait if no suitable gpu exists
                try:
                    gpu = await self.progress_mon.best_gpu()
                except Exception as e:
                    print(e)
                    raise

                # define environment -> set execution on specific GPU
                _env = dict(os.environ)
                _env['THEANO_FLAGS'] = 'base_compiledir={0},device=gpu{1}'.format(
                    comp_dir, gpu.dev)

                # make sure compiledir exists
                try:
                    os.mkdir(comp_dir)
                except FileExistsError:
                    pass

                # Start up a worker
                print('Started on ' + prefix + ' using ' + _env[
                    'THEANO_FLAGS'] + '\n')
                self.job_progress(prefix, 80, 'init')

                p = await async_subprocess('mypython3', '-i',
                                           self.basemodel_path,
                                           stdout=PIPE,
                                           stderr=PIPE,
                                           stdin=PIPE,
                                           env=_env)

                self.active_procs.append(p.p)
                stdin = p.stdin
                assert isinstance(stdin, asyncio.streams.StreamWriter)

                def wrap(lines) -> bytes:
                    cmd = b'import sys;return_code = 1;'
                    cmd += b';'.join(lines)
                    cmd += b';return_code = 0\n'
                    return cmd

                lines = [b'print("setting opts")']
                # send commands to worker to change features
                for feature_name, value in overrides:
                    lines.append('obj.opt.{0} = {1}'.format(feature_name,
                                                            value).encode())
                lines.append(b'sys.stdout.write(str(obj.opt))')
                lines.append(b'del obj.recipe.saved_params')
                # startup the training
                lines.append(b'obj.cook.auto_train()')
                lines.append('obj.save("{0}")'.format(fname).encode())


                # wrap lines to one statement and feed to process
                cmd = wrap(lines)
                stdin.write(cmd)

                # call sys exit so process will terminate
                stdin.write(b'print("return_code: ", return_code)\n')
                stdin.write(b'sys.exit(return_code)\n')
                await stdin.drain()

                # wait some time for the worker to actually start using the
                # GPU before releasing startup lock
                for i in range(40):
                    await asyncio.sleep(1)
                    self.job_progress(prefix, 60 + i, 'init')
                # startup lock released here

            # write to progress_monitor and log - wait for process
            with open(out_dir + os.sep + prefix + '.log', 'w',
                      buffering=1) as logfile:
                logfile.write(cmd.decode())
                logfile.flush()

                # noinspection PyTypeChecker
                to_do = [self.write_to_log(p.stdout, logfile, prefix),
                         self.write_to_log(p.stderr, logfile, prefix),
                         p.p.wait()]
                await asyncio.gather(*to_do, loop=self.loop)

            # process terminated. remove process from active list
            # and determine if successful
            self.active_procs.remove(p.p)
            if p.p.returncode == 0:
                self.job_progress(prefix, 100, 'complete')
                print('Completed ' + fname)
            else:
                # return code NOT 0 - something has gone wrong
                self.job_progress(prefix, 100, 'dead')
                warnings.warn(
                    'job "{0}" ended with returncode {1}'.format(prefix,
                                                                 p.p.returncode))

    def __del__(self):
        self.terminate()


class AsyncHCLearningRate(LearningRateMixin, AsyncHeadChef):
    pass


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
