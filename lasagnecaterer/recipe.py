"""
Instructions on how to put together different kinds of lasagnas
"""
# builtins
from collections import (namedtuple, defaultdict, OrderedDict)
from functools import (lru_cache, partial)
from weakref import WeakKeyDictionary

# pip packages
from itertools import chain

import sys
from theano import tensor as T
import theano
import lasagne as L
import numpy as np

# github packages
from elymetaclasses.events import ChainedProps, args_from_opt
from .utils import ChainPropsABCMetaclass
from .fridge import ClassSaveLoadMixin, SaveLoadZipFilemixin


class ScalarParameter:
    def opt_callback(self, instance, opt_name, value):
        self.__set__(instance, value)

    def __init__(self, opt_name, spec_type=L.init.Constant, shape=None,
                 default=0.0):
        self.opt_name = opt_name
        self.spec_type = spec_type
        self.shape = shape
        self.default = default
        self.instances = WeakKeyDictionary()

    def make_param(self, instance, set_val=None):
        opt = instance.opt
        callback = partial(self.opt_callback, instance)
        init_val = set_val if set_val is not None else opt.get(self.opt_name,
                                                            self.default)
        if not self.shape:
            param = theano.shared(self.spec_type(init_val).sample((1,))[0])
        else:
            spec = self.spec_type()
            param = L.utils.create_param(spec, shape=self.shape,
                                         name=self.opt_name)
        opt.set_callback(self.opt_name, callback)
        return param, callback

    def __get__(self, instance: ChainedProps, obj_type):
        if instance not in self.instances:
            self.instances[instance] = self.make_param(instance)
        return self.instances[instance][0]

    def __set__(self, instance, value):
        if instance not in self.instances:
            self.instances[instance] = self.make_param(instance)
        self.instances[instance][0].set_value(value)


class OneHotLayer(L.layers.Layer):
    def __init__(self, incoming, axis=0, name=None):
        self.axis = axis
        super().__init__(incoming, name)

    def get_output_for(self, input, **kwargs):
        return L.utils.one_hot(T.argmax(input, axis=self.axis),
                               input.shape[self.axis])


class LasagneBase(ChainedProps, ClassSaveLoadMixin,
                  metaclass=ChainPropsABCMetaclass):
    def __init__(self, *args, **kwargs):
        super(LasagneBase, self).__init__(*args, **kwargs)
        self.saved_params = None

    @property
    def l_in(self, features, win_sz=1):
        """
        Input layer into which x feeds
        x is [batch_sz, seq_len, win_sz, features]
        :param seq_len:
        :param features:
        :param win_sz:
        :return:
        """

        return L.layers.InputLayer((None, None, win_sz, features))

    @property
    def target_values(self):
        """
        Shared variable to hold y
        override if target values is *not* a 3D tensor
        unlikely if using lasagne
        :return:
        """
        return T.tensor3('target_output')

    @property
    def l_bottom(self):
        """
        Bottom most non-input layer
        :return:
        """
        return self.l_in

    @property
    def l_top(self):
        """
        Topmost non-output layer
        :return:
        """
        return self.l_bottom

    @property
    def l_out_flat(self):
        """
        Flattened output layer.
        output of this layer should be transformed so it matches target values
        (eg. softmax into classes)
        :return:
        """

        return self.out_transform(self.l_top)

    @property
    def l_out(self, seq_len, features):
        """
        Reshaped output layer that matches the shape of y
        :param seq_len:
        :param features:
        :return:
        """

        return L.layers.ReshapeLayer(self.l_out_flat, (-1, seq_len, features))

    def reset_params(self):
        self.init_params(None)

    def init_params(self, saved_params):
        if saved_params:
            self.set_all_params(saved_params)

    @property
    def all_params(self):
        """
        list of shared variables which hold all parameters in the lasagne
        :return:
        """
        self.init_params(self.saved_params)
        return L.layers.get_all_params(self.l_out_flat)

    @property
    def all_train_params(self):
        self.init_params(self.saved_params)
        return L.layers.get_all_params(self.l_out_flat, trainable=True)

    def set_all_params(self, params):
        params = [np.array(param, dtype=np.float32) for param in params]
        L.layers.set_all_param_values(self.l_out_flat, params)
        self.saved_params = params

    def get_all_params_copy(self):
        return L.layers.get_all_param_values(self.l_out_flat)

    @property
    def cost(self):
        """
        Shared variable with the cost of target values vs predicted values
        :param:
        :return:
        """
        flattened_output = L.layers.get_output(self.l_out_flat)
        return self.cost_metric(flattened_output)

    @property
    def cost_det(self):
        """
        Shared variable with the cost of target values vs predicted values
        Computed deterministic such that any dropout is ignored
        :param features:
        :return:
        """
        flattened_output = L.layers.get_output(self.l_out_flat,
                                               deterministic=True)
        return self.cost_metric(flattened_output)

    def compiled_function(self, *args, givens=tuple(), **kwargs):
        kwargs['givens'] = list(givens)# + [self.learning_rate]
        return theano.function(*args, **kwargs)

    learning_rate = ScalarParameter('learning_rate', default=.002)

    @property
    def train_updates(self):
        return L.updates.rmsprop(self.cost, self.all_train_params,
                                 self.learning_rate)

    @property
    def f_train(self):
        """
        Compiled theano function that takes (x, y) and trains the lasagne
        Updates use default cost (e.g with dropout)
        But f_train returns the deterministic cost
        :return:
        """

        return self.compiled_function([self.l_in.input_var, self.target_values],
                                      self.cost_det,
                                      updates=self.train_updates,
                                      allow_input_downcast=True)

    @property
    def f_train_no_return(self):
        """
        Compiled theano function that takes (x, y) and trains the lasagne
        Updates use default cost (e.g with dropout)
        Does *not* return a cost.
        :return: None
        """
        return self.compiled_function([self.l_in.input_var, self.target_values],
                                      updates=self.train_updates,
                                      allow_input_downcast=True)

    @property
    def f_cost(self):
        """
        Compiled theano function that takes (x, y) and return cost.
        No updates is made
        :return:
        """
        self.all_train_params
        return self.compiled_function([self.l_in.input_var, self.target_values],
                                      self.cost_det, allow_input_downcast=True)

    @property
    def f_predict(self, features):
        """
        Compiled theano function that takes (x) and predicts y
        Computed *deterministic*
        :return:
        """
        self.all_train_params
        resh = L.layers.ReshapeLayer(self.l_out_flat, (-1, features))
        output_transformed = self.predict_transform(resh)
        prediction = L.layers.get_output(output_transformed, deterministic=True)
        return self.compiled_function([self.l_in.input_var], prediction,
                                      allow_input_downcast=True)

    @property
    def f_predict_single(self, features):
        """
        take input sequence of arbitrary length and predict n next characters
        by feeding characters back into network
        :return:
        """
        print('compiling')
        self.all_train_params
        # make list of outputs in 1D list
        resh = L.layers.ReshapeLayer(self.l_out_flat, (-1, features))

        # Take last prediction
        y_lay = L.layers.SliceLayer(resh, -1, axis=0)

        x_next_fuzzy_lay = L.layers.ReshapeLayer(y_lay, (1, features))
        x_next_lay = self.predict_transform(x_next_fuzzy_lay)
        x_next = L.layers.get_output(x_next_lay, deterministic=True)
        return self.compiled_function([self.l_in.input_var], x_next)

    @args_from_opt(2)
    def auto_predict(self, x_next, n, features, sentinels: list = None):
        for _ in range(n):
            x_next = x_next.reshape(1, -1, 1, features)
            # self.reset_hidden_states(batched=False)
            x_next = self.f_predict_single(x_next)
            yield x_next
            if sentinels is not None:
                for sentinel in sentinels:
                    for e, ee in zip(x_next.flatten(), sentinel.flatten()):
                        if e != ee:
                            break
                    else:
                        return
        return
        outputs_info = [self.l_in.input_var]
        if hasattr(self, 'hidden_state_updates'):
            initial, computed = tuple(zip(*self.hidden_state_updates.values()))

            def auropred(*args):
                return tuple(chain([x_next], computed))

            outputs_info.extend(initial)
        else:
            def auropred(*args):
                return x_next
        n_stepVar = T.iscalar()

        (out_list, *_), updates = theano.scan(auropred,
                                              outputs_info=outputs_info,
                                              n_steps=n_stepVar)

        f = self.compiled_function([self.l_in.input_var, n_stepVar], out_list,
                                   updates=updates)

        def compiled_function_wrapper(x: np.matrix, n):
            x = x.reshape((1, -1, 1, features))
            return f(x, n)

        return compiled_function_wrapper

    def out_transform(self, l, *args):
        """
        apply transformation from outermost layer to vector of classes
        :param l:
        :param args:
        :return:
        """
        raise NotImplementedError

    def predict_transform(self, l_top: L.layers.Layer, l_out_flat: L.layers.Layer) -> L.layers.Layer:
        raise NotImplementedError

    def cost_metric(self, flattened_output, features):
        """
        apply a metric of cost to flattened output with given number of features
        :param flattened_output:
        :return:
        """
        raise NotImplementedError


class LSTMBase(LasagneBase):
    @args_from_opt(1)
    def build_recurrent_layers(self, l_prev, n_hid_lay):
        """
        Construct a number of LSTM layers taking l_prev as bottom input
        :param l_prev:
        :param n_hid_lay:
        :return:
        """
        for i in range(n_hid_lay):
            print('LSTM-in', i, l_prev.output_shape)
            l_prev = self.make_recurrent_layer(l_prev)
        return l_prev

    @args_from_opt(1)
    def make_recurrent_layer(self, l_prev, n_hid_unit, grad_clip=5):
        return L.layers.LSTMLayer(l_prev, n_hid_unit,
                                  grad_clipping=grad_clip,
                                  nonlinearity=L.nonlinearities.tanh)

    @args_from_opt(1)
    def out_transform(self, l_prev, n_hid_unit, features):
        """
        Apply non-linear transform to l_prev
        :param l_prev:
        :param n_hid_unit:
        :param features:
        :return:
        """
        l_shp = L.layers.ReshapeLayer(l_prev, (-1, n_hid_unit))
        return L.layers.DenseLayer(l_shp, num_units=features,
                                   nonlinearity=L.nonlinearities.softmax)

    @property
    def l_top(self, n_hid_unit, n_hid_lay):
        """
        fit LSTM layers in between bottom and top
        :return:
        """
        return self.build_recurrent_layers(self.l_bottom)

    @args_from_opt(1)
    def init_params(self, saved_params, W_range=0.08):
        """
        initialize all parameters to uniform in +-W_range
        :param W_range:
        :return:
        """
        if saved_params:
            super().init_params(saved_params)
        else:
            w_init = L.init.Uniform(range=W_range)
            new_params = [w_init.sample(w.shape) for w
                          in self.get_all_params_copy()]

            self.set_all_params(new_params)

    cost_stability = ScalarParameter('cost_stability', default=1e-6)

    @args_from_opt(1)
    def cost_metric(self, flattened_output, features):
        """
        Cross entropy cost metric
        :param flattened_output:
        :param features:
        :return:
        """
        normed_output = flattened_output * (1 - 2 * self.cost_stability) + self.cost_stability

        return T.nnet.categorical_crossentropy(normed_output,
                                               self.target_values.reshape(
                                                   (-1, features))).mean()

    def compiled_function(self, *args, givens=tuple(), **kwargs):
        kwargs['givens'] = list(givens)# + [self.cost_stability]
        return super(LSTMBase, self).compiled_function(*args, **kwargs)

    @args_from_opt(2)
    def predict_transform(self, l_top: L.layers.Layer,
                          l_out_flat: L.layers.DenseLayer, n_hid_unit) -> L.layers.Layer:
        return OneHotLayer(l_out_flat, axis=1, name='onehot_pred')


class GRUBase(LasagneBase):
    class OneHotLayer(L.layers.Layer):
        def __init__(self, incoming, axis=0, name=None):
            self.axis = axis
            super().__init__(incoming, name)

        def get_output_for(self, input, **kwargs):
            return L.utils.one_hot(T.argmax(input, axis=self.axis),
                                   input.shape[self.axis])

    @args_from_opt(1)
    def build_recurrent_layers(self, l_prev, n_hid_lay):
        """
        Construct a number of LSTM layers taking l_prev as bottom input
        :param l_prev:
        :param n_hid_lay:
        :return:
        """
        for i in range(n_hid_lay):
            print('GRU-in', i, l_prev.output_shape)
            l_prev = self.make_recurrent_layer(l_prev)
        return l_prev

    @args_from_opt(1)
    def make_recurrent_layer(self, l_prev, n_hid_unit, grad_clip=5):
        return L.layers.GRULayer(l_prev, n_hid_unit,
                                  grad_clipping=grad_clip,
                                  nonlinearity=L.nonlinearities.tanh)

    @args_from_opt(1)
    def out_transform(self, l_prev, n_hid_unit, features):
        """
        Apply non-linear transform to l_prev
        :param l_prev:
        :param n_hid_unit:
        :param features:
        :return:
        """
        l_shp = L.layers.ReshapeLayer(l_prev, (-1, n_hid_unit))
        return L.layers.DenseLayer(l_shp, num_units=features,
                                   nonlinearity=L.nonlinearities.softmax)

    @property
    def l_top(self):
        """
        fit LSTM layers in between bottom and top
        :return:
        """
        return self.build_recurrent_layers(self.l_bottom)

    @args_from_opt(1)
    def init_params(self, saved_params, W_range=0.08):
        """
        initialize all parameters to uniform in +-W_range
        :param W_range:
        :return:
        """
        if saved_params:
            super().init_params(saved_params)
        else:
            w_init = L.init.Uniform(range=W_range)
            new_params = [w_init.sample(w.shape) for w
                          in self.get_all_params_copy()]

            self.set_all_params(new_params)

    @args_from_opt(1)
    def cost_metric(self, flattened_output, features):
        """
        Cross entropy cost metric
        :param flattened_output:
        :param features:
        :return:
        """
        return T.nnet.categorical_crossentropy(flattened_output,
                                               self.target_values.reshape(
                                                   (-1, features))).mean()

    @args_from_opt(2)
    def predict_transform(self, l_top: L.layers.Layer,
                          l_out_flat: L.layers.DenseLayer, n_hid_unit) -> L.layers.Layer:
        return OneHotLayer(l_out_flat, axis=1, name='onehot_pred')


class MixinBase(metaclass=ChainPropsABCMetaclass): pass


class RandomPredictMixin(MixinBase):
    pred_sigma = ScalarParameter('pred_sigma')

    class ScaledSoftmax:
        def __init__(self, scale=1):
            self.scale = scale

        def __call__(self, x):
            return T.nnet.softmax(x * self.scale)

    class RandomOnehotLayer(L.layers.GaussianNoiseLayer):
        def get_output_for(self, input, probabilistic=True, **kwargs):
            if probabilistic:
                if self.sigma:
                    input = T.pow(input, self.sigma)  # boost
                csum = T.cumsum(input, axis=1)
                last_col = T.repeat(csum[:, -1:None], csum.shape[1], axis=1)
                normed = csum / last_col
                normed = T.concatenate([T.zeros((csum.shape[0], 1)), normed],
                                       axis=1)
                rand = self._srng.uniform(input.shape[0:1]).reshape((-1, 1))
                idx = T.repeat(rand, input.shape[1], axis=1)

                picked = T.le(normed[:, :-1], idx) * T.gt(normed[:, 1:], idx)
                return picked
            else:
                return L.utils.one_hot(T.argmax(input, axis=1), input.shape[1])

    @args_from_opt(1)
    def predict_transform(self, l_top, pred_sigma=0):
        return self.RandomOnehotLayer(l_out, sigma=self.pred_sigma,
                                      name='pred_trans')


class DebugMixin(MixinBase):
    rho = ScalarParameter('rho', default=0.9)
    epsilon = ScalarParameter('epsilon', default=1e-6)

    def update_debug(self, loss, params):
        grads = L.updates.get_or_compute_grads(loss, params)
        debug_info = OrderedDict()
        learning_rate = self.learning_rate
        rho = self.rho
        epsilon = self.epsilon
        init_params = OrderedDict(rho=rho, epsilon=epsilon, leaning_rate=learning_rate,
                                  loss=loss)
        debug_info['init_params'] = init_params
        for param, grad in zip(params, grads):
            calculations = OrderedDict()

            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            calculations['accu'] = accu
            calculations['rho * accu'] = rho * accu
            calculations['(1 - rho) * grad ** 2'] = (1 - rho) * grad ** 2

            calculations['accu_new'] = rho * accu + (1 - rho) * grad ** 2
            calculations['param'] = param
            calculations['grad'] = grad
            calculations['accu_new + epsilon'] = calculations['accu_new'] + epsilon
            calculations['T.sqrt(accu_new + epsilon)'] = T.sqrt(calculations['accu_new'] + epsilon)
            calculations['new_param'] = param - (learning_rate * grad /
                                      T.sqrt(calculations['accu_new'] + epsilon))
            debug_info[param] = calculations

        return debug_info

    def f_train_hidden_states(self):
        return self.compiled_function([self.l_in.input_var, self.target_values],
                                      [self.cost_det],
                                      updates=self.train_updates,
                                      allow_input_downcast=True,
                                      )


    def unpack_debug_info(self, debug_info, computed=tuple()):
        computed = list(computed)
        computed.reverse()
        for key, value in debug_info.items():
            if computed:
                name = key.name if hasattr(key, 'name') else str(key)
                print('#' * 10 + name + '#'*10)
            for kkey, vvalue in value.items():
                if computed:
                    nname = kkey.name if hasattr(kkey, 'name') else str(kkey)
                    print('-'*10 + nname + '-'*10)
                    print(computed.pop())
                    print('-'*20)
                else:
                    yield vvalue

    @property
    def debug_info(self):
        return self.update_debug(self.cost, self.all_train_params)

    @staticmethod
    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                    (np.isnan(output[0]).any() or np.isinf(output[0]).any())):
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                raise ZeroDivisionError


    @property
    def f_train_debug(self):
        values = list(self.unpack_debug_info(self.debug_info))
        return self.compiled_function([self.l_in.input_var, self.target_values],
                                      [self.cost_det, self.cost] + values,
                                      updates=self.train_updates,
                                      allow_input_downcast=True,
                                      )
    @property
    def f_train_detect_nan(self):
        from theano.compile.nanguardmode import NanGuardMode
        return self.compiled_function([self.l_in.input_var, self.target_values],
                                [self.cost_det], updates=self.train_updates,
                               mode=NanGuardMode(True, True, True))


    @property
    def f_grad_detect_nan(self):
        grads = L.updates.get_or_compute_grads(self.cost, self.all_train_params)
        return self.compiled_function([self.l_in.input_var, self.target_values],
                               grads + [self.cost_det, self.cost, L.layers.get_output(self.l_out)],
                               mode=theano.compile.MonitorMode(
                                      post_func=self.detect_nan))

class SSEMixin(MixinBase):
    @args_from_opt(1)
    def cost_metric(self, flattened_output, features):
        """
        Cross entropy cost metric
        :param flattened_output:
        :param features:
        :return:
        """
        diff = flattened_output - self.target_values.reshape((-1, features))
        return (diff @ diff.T).mean()

    @args_from_opt(1)
    def out_transform(self, l_prev, n_hid_unit, features):
        """
        Apply non-linear transform to l_prev
        :param l_prev:
        :param n_hid_unit:
        :param features:
        :return:
        """

        l_shp = L.layers.ReshapeLayer(l_prev, (-1, n_hid_unit))
        l_cat = L.layers.ConcatLayer(
            [l_shp, L.layers.ReshapeLayer(self.l_in, (-1, features))])
        return L.layers.DenseLayer(l_cat, num_units=features)

    def predict_transform(self, l_out) -> L.layers.Layer:
        return l_out


class StateReuseMixin(MixinBase):
    class OutputSplitLayer(L.layers.Layer):
        def __init__(self, input_layer: L.layers.MergeLayer, name=None):
            super().__init__(input_layer, name=name)
            self.input_layer = input_layer

        def get_output_shape_for(self, input_shape):
            return input_shape

        def get_output_for(self, input, **kwargs):
            hid_out, *_ = input

            # When it is requested that we only return the final sequence step,
            # we need to slice it out immediately after scan is applied
            if self.input_layer.only_return_final:
                hid_out = hid_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                hid_out = hid_out.dimshuffle(1, 0, 2)

                # if scan is backward reverse the output
                if self.input_layer.backwards:
                    hid_out = hid_out[:, ::-1]
            return hid_out

    class ExtractLastStates(L.layers.Layer):
        def __init__(self, input_layer: L.layers.MergeLayer, name=None):
            super().__init__(input_layer, name=name)
            self.input_layer = input_layer

        def get_output_for(self, input, **kwargs):
            if self.input_layer.backwards:
                return tuple(state[0] for state in input)
            return tuple(state[-1] for state in input)

    @property
    def hidden_state_updates(self):
        return WeakKeyDictionary()

    @args_from_opt('batched')
    def reset_hidden_states(self, batched=True, batch_sz=None, n_hid_unit=None):
        if batched:
            zeros = np.zeros((batch_sz, n_hid_unit), dtype=np.float32)
        else:
            zeros = np.zeros((1, n_hid_unit), dtype=np.float32)

        for layer, (param, update) in self.hidden_state_updates.items():
            param.set_value(zeros)

    class HiddenStates(OrderedDict, SaveLoadZipFilemixin):
        def __iter__(self):
            return self.keys()

        def from_dict(cls, *args, **kwargs):
            return cls(kwargs)

        def to_dict(self) -> OrderedDict:
            return self

        def bootstrap(self):
            pass

        @classmethod
        def from_weak_dict(cls, hidden_states: WeakKeyDictionary):
            return cls(
                (id(layer), param.get_value()) for layer, (param, update) in
                hidden_states.items())

    def set_hidden_state(self, checkpoint: HiddenStates):
        for layer, (param, update) in self.hidden_state_updates.items():
            param.set_value(checkpoint[id(layer)])

    def get_hidden_state(self) -> HiddenStates:
        return self.HiddenStates.from_weak_dict(self.hidden_state_updates)

    @property
    def f_print_hidden(self):

        states = L.layers.get_output(list(self.hidden_state_updates.keys()))
        return theano.function(list(), outputs=states)
        # for layer, (param, update) in zip(self.hidden_state_updates.items()):

    @args_from_opt(1)
    def make_recurrent_layer(self, l_prev, batch_sz, seq_len, n_hid_unit,
                             unroll=False):
        state_shape = (batch_sz, n_hid_unit)
        state_params = [L.utils.create_param(L.init.Constant(0.0), state_shape,
                                       name=name + '_param') for name in self.state_names]

        state_in_layers = [L.layers.InputLayer((batch_sz, n_hid_unit),
                                             input_var=param, name=name + '_lay')
                         for param, name in zip(state_params, self.state_names)]

        if unroll:
            sys.setrecursionlimit(int(2000 + batch_sz * seq_len))
        raw_lay = self._make_recurrent_layer(l_prev, state_in_layers)
        return_lay = self.OutputSplitLayer(raw_lay, name='return')
        last_state_lay = self.ExtractLastStates(raw_lay, name='last_states')

        last_states = L.layers.get_output(last_state_lay)
        for _in, last, param in zip(state_in_layers, last_states, state_params):
            self.hidden_state_updates[_in] = (param, last)
        return return_lay

    def _make_recurrent_layer(self, l_prev, state_layers):
        raise NotImplementedError

    @property
    def state_names(self) -> tuple:
        raise NotImplementedError

    def compiled_function(self, *args, updates=tuple(), **kwargs):

        if isinstance(updates, dict):
            updates = updates.items()

        kwargs['updates'] = list(updates) + list(self.hidden_state_updates.values())
        return super().compiled_function(*args, **kwargs)


class LSTMStateReuseMixin(StateReuseMixin):
    @property
    def state_names(self) -> tuple:
        return 'cell', 'hidden'

    @args_from_opt(2)
    def _make_recurrent_layer(self, l_prev, state_layers, n_hid_unit,
                              grad_clip=5, unroll=False):
        cell_lay, hid_lay = state_layers
        raw_lay = self.LSTMLayer(l_prev, n_hid_unit,
                                 hid_init=hid_lay, cell_init=cell_lay,
                                 grad_clipping=grad_clip,
                                 nonlinearity=L.nonlinearities.tanh,
                                 name='LSTM_raw',
                                 unroll_scan=unroll)
        return raw_lay

    class LSTMLayer(L.layers.LSTMLayer):
        """
        Copy of standard lasagne.layers.LSTMLayer
        with overwritten get_output_for such that it returns cell state
        """

        def get_output_for(self, inputs, **kwargs):
            """
            Compute this layer's output function given a symbolic input variable



            Parameters
            ----------
            inputs : list of theano.TensorType
                `inputs[0]` should always be the symbolic input variable.  When
                this layer has a mask input (i.e. was instantiated with
                `mask_input != None`, indicating that the lengths of sequences in
                each batch vary), `inputs` should have length 2, where `inputs[1]`
                is the `mask`.  The `mask` should be supplied as a Theano variable
                denoting whether each time step in each sequence in the batch is
                part of the sequence or not.  `mask` should be a matrix of shape
                ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
                (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
                of sequence i)``. When the hidden state of this layer is to be
                pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
                should have length at least 2, and `inputs[-1]` is the hidden state
                to prefill with. When the cell state of this layer is to be
                pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
                should have length at least 2, and `inputs[-1]` is the hidden state
                to prefill with. When both the cell state and the hidden state are
                being pre-filled `inputs[-2]` is the hidden state, while
                `inputs[-1]` is the cell state.

            Returns
            -------
            layer_output : theano.TensorType
                Symbolic output variable.
            """
            # a couple of local imports
            unroll_scan = L.utils.unroll_scan
            Layer = L.layers.Layer

            # Retrieve the layer input
            input = inputs[0]
            # Retrieve the mask when it is supplied
            mask = None
            hid_init = None
            cell_init = None
            if self.mask_incoming_index > 0:
                mask = inputs[self.mask_incoming_index]
            if self.hid_init_incoming_index > 0:
                hid_init = inputs[self.hid_init_incoming_index]
            if self.cell_init_incoming_index > 0:
                cell_init = inputs[self.cell_init_incoming_index]

            # Treat all dimensions after the second as flattened feature dimensions
            if input.ndim > 3:
                input = T.flatten(input, 3)

            # Because scan iterates over the first dimension we dimshuffle to
            # (n_time_steps, n_batch, n_features)
            input = input.dimshuffle(1, 0, 2)
            seq_len, num_batch, _ = input.shape

            # Stack input weight matrices into a (num_inputs, 4*num_units)
            # matrix, which speeds up computation
            W_in_stacked = T.concatenate(
                [self.W_in_to_ingate, self.W_in_to_forgetgate,
                 self.W_in_to_cell, self.W_in_to_outgate], axis=1)

            # Same for hidden weight matrices
            W_hid_stacked = T.concatenate(
                [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
                 self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

            # Stack biases into a (4*num_units) vector
            b_stacked = T.concatenate(
                [self.b_ingate, self.b_forgetgate,
                 self.b_cell, self.b_outgate], axis=0)

            if self.precompute_input:
                # Because the input is given for all time steps, we can
                # precompute_input the inputs dot weight matrices before scanning.
                # W_in_stacked is (n_features, 4*num_units). input is then
                # (n_time_steps, n_batch, 4*num_units).
                input = T.dot(input, W_in_stacked) + b_stacked

            # At each call to scan, input_n will be (n_batch, 4*num_units)
            # such that it is a slice across all sequences in the batch representing
            # the input values for a time step
            # We define a slicing function that extract the input to each LSTM gate
            def slice_w(x, n):
                return x[:, n * self.num_units:(n + 1) * self.num_units]

            # Create single recurrent computation step function
            # input_n is the n'th vector of the input
            def step(input_n, cell_previous, hid_previous, *args):
                if not self.precompute_input:
                    input_n = T.dot(input_n, W_in_stacked) + b_stacked

                # Calculate gates pre-activations and slice
                gates = input_n + T.dot(hid_previous, W_hid_stacked)

                # Clip gradients
                if self.grad_clipping:
                    gates = theano.gradient.grad_clip(
                        gates, -self.grad_clipping, self.grad_clipping)

                # Extract the pre-activation gate values
                ingate = slice_w(gates, 0)
                forgetgate = slice_w(gates, 1)
                cell_input = slice_w(gates, 2)
                outgate = slice_w(gates, 3)

                if self.peepholes:
                    # Compute peephole connections
                    ingate += cell_previous * self.W_cell_to_ingate
                    forgetgate += cell_previous * self.W_cell_to_forgetgate

                # Apply nonlinearities
                ingate = self.nonlinearity_ingate(ingate)
                forgetgate = self.nonlinearity_forgetgate(forgetgate)
                cell_input = self.nonlinearity_cell(cell_input)

                # Compute new cell value
                cell = forgetgate * cell_previous + ingate * cell_input

                if self.peepholes:
                    outgate += cell * self.W_cell_to_outgate
                outgate = self.nonlinearity_outgate(outgate)

                # Compute new hidden unit activation
                hid = outgate * self.nonlinearity(cell)
                return [cell, hid]

            def step_masked(input_n, mask_n, cell_previous, hid_previous,
                            *args):
                cell, hid = step(input_n, cell_previous, hid_previous, *args)

                # Skip over any input with mask 0 by copying the previous
                # hidden state; proceed normally for any input with mask 1.
                cell = T.switch(mask_n, cell, cell_previous)
                hid = T.switch(mask_n, hid, hid_previous)

                return [cell, hid]

            if mask is not None:
                # mask is given as (batch_size, seq_len). Because scan iterates
                # over first dimension, we dimshuffle to (seq_len, batch_size) and
                # add a broadcastable dimension
                mask = mask.dimshuffle(1, 0, 'x')
                sequences = [input, mask]
                step_fun = step_masked
            else:
                sequences = input
                step_fun = step

            ones = T.ones((num_batch, 1))
            if not isinstance(self.cell_init, Layer):
                # Dot against a 1s vector to repeat to shape (num_batch, num_units)
                cell_init = T.dot(ones, self.cell_init)

            if not isinstance(self.hid_init, Layer):
                # Dot against a 1s vector to repeat to shape (num_batch, num_units)
                hid_init = T.dot(ones, self.hid_init)

            # The hidden-to-hidden weight matrix is always used in step
            non_seqs = [W_hid_stacked]
            # The "peephole" weight matrices are only used when self.peepholes=True
            if self.peepholes:
                non_seqs += [self.W_cell_to_ingate,
                             self.W_cell_to_forgetgate,
                             self.W_cell_to_outgate]

            # When we aren't precomputing the input outside of scan, we need to
            # provide the input weights and biases to the step function
            if not self.precompute_input:
                non_seqs += [W_in_stacked, b_stacked]

            if self.unroll_scan:
                # Retrieve the dimensionality of the incoming layer
                input_shape = self.input_shapes[0]
                # Explicitly unroll the recurrence instead of using scan
                cell_out, hid_out = unroll_scan(
                    fn=step_fun,
                    sequences=sequences,
                    outputs_info=[cell_init, hid_init],
                    go_backwards=self.backwards,
                    non_sequences=non_seqs,
                    n_steps=input_shape[1])
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                cell_out, hid_out = theano.scan(
                    fn=step_fun,
                    sequences=sequences,
                    outputs_info=[cell_init, hid_init],
                    go_backwards=self.backwards,
                    truncate_gradient=self.gradient_steps,
                    non_sequences=non_seqs,
                    strict=True)[0]
            return hid_out, cell_out


class GRUStateReuseMixin(StateReuseMixin):
    @property
    def state_names(self) -> tuple:
        return 'hidden'

    @args_from_opt(2)
    def _make_recurrent_layer(self, l_prev, state_layers, n_hid_unit,
                              grad_clip=100, unroll=False):
        hid_lay = state_layers[0]
        raw_lay = self.GRULayer(l_prev, n_hid_unit,
                                 hid_init=hid_lay,
                                 grad_clipping=grad_clip,
                                 nonlinearity=L.nonlinearities.tanh,
                                 name='GRU_raw',
                                 unroll_scan=unroll)
        return raw_lay

    class GRULayer(L.layers.GRULayer):
        """
        Copy of standard lasagne.layers.LSTMLayer
        with overwritten get_output_for such that it returns cell state
        """

        def get_output_for(self, inputs, **kwargs):
            """
            Compute this layer's output function given a symbolic input variable

            Parameters
            ----------
            inputs : list of theano.TensorType
                `inputs[0]` should always be the symbolic input variable.  When
                this layer has a mask input (i.e. was instantiated with
                `mask_input != None`, indicating that the lengths of sequences in
                each batch vary), `inputs` should have length 2, where `inputs[1]`
                is the `mask`.  The `mask` should be supplied as a Theano variable
                denoting whether each time step in each sequence in the batch is
                part of the sequence or not.  `mask` should be a matrix of shape
                ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
                (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
                of sequence i)``. When the hidden state of this layer is to be
                pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
                should have length at least 2, and `inputs[-1]` is the hidden state
                to prefill with.

            Returns
            -------
            layer_output : theano.TensorType
                Symbolic output variable.
            """
            unroll_scan = L.utils.unroll_scan
            Layer = L.layers.Layer

            # Retrieve the layer input
            input = inputs[0]
            # Retrieve the mask when it is supplied
            mask = None
            hid_init = None
            if self.mask_incoming_index > 0:
                mask = inputs[self.mask_incoming_index]
            if self.hid_init_incoming_index > 0:
                hid_init = inputs[self.hid_init_incoming_index]

            # Treat all dimensions after the second as flattened feature dimensions
            if input.ndim > 3:
                input = T.flatten(input, 3)

            # Because scan iterates over the first dimension we dimshuffle to
            # (n_time_steps, n_batch, n_features)
            input = input.dimshuffle(1, 0, 2)
            seq_len, num_batch, _ = input.shape

            # Stack input weight matrices into a (num_inputs, 3*num_units)
            # matrix, which speeds up computation
            W_in_stacked = T.concatenate(
                [self.W_in_to_resetgate, self.W_in_to_updategate,
                 self.W_in_to_hidden_update], axis=1)

            # Same for hidden weight matrices
            W_hid_stacked = T.concatenate(
                [self.W_hid_to_resetgate, self.W_hid_to_updategate,
                 self.W_hid_to_hidden_update], axis=1)

            # Stack gate biases into a (3*num_units) vector
            b_stacked = T.concatenate(
                [self.b_resetgate, self.b_updategate,
                 self.b_hidden_update], axis=0)

            if self.precompute_input:
                # precompute_input inputs*W. W_in is (n_features, 3*num_units).
                # input is then (n_batch, n_time_steps, 3*num_units).
                input = T.dot(input, W_in_stacked) + b_stacked

            # At each call to scan, input_n will be (n_time_steps, 3*num_units).
            # We define a slicing function that extract the input to each GRU gate
            def slice_w(x, n):
                return x[:, n*self.num_units:(n+1)*self.num_units]

            # Create single recurrent computation step function
            # input__n is the n'th vector of the input
            def step(input_n, hid_previous, *args):
                # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
                hid_input = T.dot(hid_previous, W_hid_stacked)

                if self.grad_clipping:
                    input_n = theano.gradient.grad_clip(
                        input_n, -self.grad_clipping, self.grad_clipping)
                    hid_input = theano.gradient.grad_clip(
                        hid_input, -self.grad_clipping, self.grad_clipping)

                if not self.precompute_input:
                    # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                    input_n = T.dot(input_n, W_in_stacked) + b_stacked

                # Reset and update gates
                resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
                updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
                resetgate = self.nonlinearity_resetgate(resetgate)
                updategate = self.nonlinearity_updategate(updategate)

                # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
                hidden_update_in = slice_w(input_n, 2)
                hidden_update_hid = slice_w(hid_input, 2)
                hidden_update = hidden_update_in + resetgate*hidden_update_hid
                if self.grad_clipping:
                    hidden_update = theano.gradient.grad_clip(
                        hidden_update, -self.grad_clipping, self.grad_clipping)
                hidden_update = self.nonlinearity_hid(hidden_update)

                # Compute (1 - u_t)h_{t - 1} + u_t c_t
                hid = (1 - updategate)*hid_previous + updategate*hidden_update
                return hid

            def step_masked(input_n, mask_n, hid_previous, *args):
                hid = step(input_n, hid_previous, *args)

                # Skip over any input with mask 0 by copying the previous
                # hidden state; proceed normally for any input with mask 1.
                hid = T.switch(mask_n, hid, hid_previous)

                return hid

            if mask is not None:
                # mask is given as (batch_size, seq_len). Because scan iterates
                # over first dimension, we dimshuffle to (seq_len, batch_size) and
                # add a broadcastable dimension
                mask = mask.dimshuffle(1, 0, 'x')
                sequences = [input, mask]
                step_fun = step_masked
            else:
                sequences = [input]
                step_fun = step

            if not isinstance(self.hid_init, Layer):
                # Dot against a 1s vector to repeat to shape (num_batch, num_units)
                hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

            # The hidden-to-hidden weight matrix is always used in step
            non_seqs = [W_hid_stacked]
            # When we aren't precomputing the input outside of scan, we need to
            # provide the input weights and biases to the step function
            if not self.precompute_input:
                non_seqs += [W_in_stacked, b_stacked]

            if self.unroll_scan:
                # Retrieve the dimensionality of the incoming layer
                input_shape = self.input_shapes[0]
                # Explicitly unroll the recurrence instead of using scan
                hid_out = unroll_scan(
                    fn=step_fun,
                    sequences=sequences,
                    outputs_info=[hid_init],
                    go_backwards=self.backwards,
                    non_sequences=non_seqs,
                    n_steps=input_shape[1])[0]
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                hid_out = theano.scan(
                    fn=step_fun,
                    sequences=sequences,
                    go_backwards=self.backwards,
                    outputs_info=[hid_init],
                    non_sequences=non_seqs,
                    truncate_gradient=self.gradient_steps,
                    strict=True)[0]

            return hid_out


class LearningRateMixin(MixinBase):
    @args_from_opt(2)
    def set_learning_rate(self, alpha=None, step=None,
                         start_alpha=(2 * 10 ** (-3)), alpha_factor=.95):
        if alpha is None:
            alpha = start_alpha * alpha_factor ** step
        self.learning_rate = alpha


class DropoutMixinBase(MixinBase):
    @args_from_opt(1)
    def apply_dropout(self, l_prev, specific_dropout=None, dropout=.5):
        if specific_dropout is None:
            return L.layers.DropoutLayer(l_prev, p=dropout)
        return L.layers.DropoutLayer(l_prev, p=specific_dropout)


class DropoutInMixin(DropoutMixinBase):
    @property
    def l_bottom(self, dropout_in=None):
        return self.apply_dropout(super().l_bottom, specific_dropout=dropout_in)


class DropoutOutMixin(DropoutMixinBase):
    @property
    def l_top(self, dropout_out=None):
        return self.apply_dropout(super().l_top, specific_dropout=dropout_out)


# noinspection PyUnresolvedReferences
class DropoutBetweenMixin(DropoutMixinBase):
    """
    Apply dropout between two consecutive recurrent layers
    """

    @args_from_opt(1)
    def build_recurrent_layers(self, l_prev, n_hid_lay, dropout_mid=None):
        """

        :param l_prev:
        :param n_hid_lay:
        :param dropout_mid:
        :return:
        """
        for i in range(n_hid_lay):
            print('Layer #', i, l_prev.output_shape)

            l_prev = self.make_recurrent_layer(l_prev)
            if i + 1 < n_hid_lay:
                l_prev = self.apply_dropout(l_prev,
                                            specific_dropout=dropout_mid)

        return l_prev
