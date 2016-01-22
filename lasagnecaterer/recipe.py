"""
Instructions on how to put together different kinds of lasagnas
"""
# builtins
from collections import (namedtuple, defaultdict)
from functools import (lru_cache, partial)

# pip packages
from theano import tensor as T
import theano
import lasagne as L

# github packages
from elymetaclasses.events import ChainedProps


class LSTMBase(ChainedProps):
    @property
    def l_in(self, seq_len, features, win_sz=1):
        return L.layers.InputLayer((None, seq_len, win_sz, features))

    @property
    def target_values(self):
        return T.tensor3('target_output')

    def build_lstm_layers(self, l_prev, n_hid_unit, n_hid_lay):
        for i in range(n_hid_lay):
            print('LSTM-in', i, l_prev.output_shape)
            l_prev = L.layers.LSTMLayer(l_prev, n_hid_unit,
                                        grad_clipping=100,
                                        nonlinearity=L.nonlinearities.tanh)
        return l_prev

    def nonlin_out(self, l_prev, n_hid_unit, features):
        l_shp = L.layers.ReshapeLayer(l_prev, (-1, n_hid_unit))
        return L.layers.DenseLayer(l_shp, num_units=features,
                                   nonlinearity=L.nonlinearities.softmax)

    @property
    def l_out_flat(self, n_hid_unit, n_hid_lay, features):
        l = self.build_lstm_layers(self.l_in, n_hid_unit, n_hid_lay)
        return self.nonlin_out(l, n_hid_unit, features)

    @property
    def l_out(self, seq_len, features):
        return L.layers.ReshapeLayer(self.l_out_flat, (-1, seq_len, features))

    @property
    def all_params(self, W_range=0.08):
        all_params = L.layers.get_all_params(self.l_out_flat)
        w_init = L.init.Uniform(range=W_range)
        for param in all_params:
            w = param.get_value()
            w[:] = w_init.sample(w.shape)
        return all_params

    @property
    def cost(self, features):
        target_values = self.target_values
        flattened_output = L.layers.get_output(self.l_out_flat)
        return T.nnet.categorical_crossentropy(flattened_output,
                                               target_values.reshape(
                                                       (-1, features))).mean()

    @property
    def cost_det(self, features):
        target_values = self.target_values
        flattened_output = L.layers.get_output(self.l_out_flat,
                                               deterministic=True)
        return T.nnet.categorical_crossentropy(flattened_output,
                                               target_values.reshape(
                                                       (-1, features))).mean()

    @property
    def f_train(self):
        updates = L.updates.rmsprop(self.cost, self.all_params)
        return theano.function([self.l_in.input_var, self.target_values],
                               self.cost,
                               updates=updates, allow_input_downcast=True)

    @property
    def f_cost(self):
        return theano.function([self.l_in.input_var, self.target_values],
                               self.cost_det, allow_input_downcast=True)

    @property
    def f_predict(self):
        output = L.layers.get_output(self.l_out, deterministic=True)
        return theano.function([self.l_in.input_var], output,
                               allow_input_downcast=True)


class LearningRateMixin(ChainedProps):
    @property
    def f_train_alpha(self, decay=.95):
        alphaVar = T.fscalar()
        updates = L.updates.rmsprop(self.cost, self.all_params, alphaVar, decay)
        return theano.function(
                [alphaVar, self.l_in.input_var, self.target_values], self.cost,
                updates=updates, allow_input_downcast=True)

    @property
    def learningopts(self, alpha=2 * 10 ** (-3), alpha_factor=.95):
        return alpha, alpha_factor

    @lru_cache()
    def f_train(self, alpha=None, step=None):
        """
        training with learninng rate
        :param alpha: the learning rate to use. None -> default
        :param step: calculate learningrate from  start_alpha * alpha_factor ** step
        :return: partial function
        """
        f = self.f_train_alpha
        if alpha is None:
            start_alpha, alpha_factor = self.learningopts
            alpha = start_alpha * alpha_factor ** step

        return partial(f, alpha)


class DropoutInOutMixin(ChainedProps):
    def apply_dropout(self, l_prev, dropout):
        return L.layers.DropoutLayer(l_prev, p=dropout)

    @property
    def l_in(self, dropout):
        return self.apply_dropout(super().l_in, dropout=.5)

    @property
    def l_out_flat(self, dropout=.5):
        return self.apply_dropout(super().l_out_flat, dropout)