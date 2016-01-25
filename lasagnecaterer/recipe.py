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
from elymetaclasses.events import ChainedProps, args_from_opt


class LasagneBase(ChainedProps):
    @property
    def l_in(self, seq_len, features, win_sz=1):
        """
        Input layer into which x feeds
        x is [batch_sz, seq_len, win_sz, features]
        :param seq_len:
        :param features:
        :param win_sz:
        :return:
        """
        return L.layers.InputLayer((None, seq_len, win_sz, features))

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
        return self.nonlin_out(self.l_top)

    @property
    def l_out(self, seq_len, features):
        """
        Reshaped output layer that matches the shape of y
        :param seq_len:
        :param features:
        :return:
        """

        return L.layers.ReshapeLayer(self.l_out_flat, (-1, seq_len, features))

    @property
    def all_params(self):
        """
        list of shared variables which hold all parameters in the lasagne
        :return:
        """
        return L.layers.get_all_params(self.l_out_flat)

    @property
    def cost(self, features):
        """
        Shared variable with the cost of target values vs predicted values
        :param features:
        :return:
        """
        flattened_output = L.layers.get_output(self.l_out_flat)
        return self.cost_metric(flattened_output, features)

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

    @property
    def f_train(self):
        """
        Compiled theano function that takes (x, y) and trains the lasagne
        :return:
        """
        updates = L.updates.rmsprop(self.cost, self.all_params)
        return theano.function([self.l_in.input_var, self.target_values],
                               self.cost,
                               updates=updates, allow_input_downcast=True)

    @property
    def f_cost(self):
        """
        Compiled theano function that takes (x, y) and return cost.
        No updates is made
        :return:
        """
        return theano.function([self.l_in.input_var, self.target_values],
                               self.cost_det, allow_input_downcast=True)

    @property
    def f_predict(self):
        """
        Compiled theano function that takes (x) and predicts y
        Computed *deterministic*
        :return:
        """
        output = L.layers.get_output(self.l_out, deterministic=True)
        return theano.function([self.l_in.input_var], output,
                               allow_input_downcast=True)

    def out_transform(self, l, *args):
        """
        apply transformation from outermost layer to vector of classes
        :param l:
        :param args:
        :return:
        """
        raise NotImplementedError

    def cost_metric(self, flattened_output, features):
        """
        apply a metric of cost to flattened output with given number of features
        :param flattened_output:
        :param features:
        :return:
        """
        raise NotImplementedError


class LSTMBase(LasagneBase):

    @args_from_opt(1)
    def build_lstm_layers(self, l_prev, n_hid_unit, n_hid_lay):
        """
        Construct a number of LSTM layers taking l_prev as boottom input
        :param l_prev:
        :param n_hid_unit:
        :param n_hid_lay:
        :return:
        """
        for i in range(n_hid_lay):
            print('LSTM-in', i, l_prev.output_shape)
            l_prev = L.layers.LSTMLayer(l_prev, n_hid_unit,
                                        grad_clipping=100,
                                        nonlinearity=L.nonlinearities.tanh)
        return l_prev

    @args_from_opt(1)
    def nonlin_out(self, l_prev, n_hid_unit, features):
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
        return self.build_lstm_layers(self.l_bottom)

    @property
    def l_out_flat(self):
        return self.nonlin_out(self.l_top)

    @property
    def l_out(self, seq_len, features):
        return L.layers.ReshapeLayer(self.l_out_flat, (-1, seq_len, features))

    @property
    def all_params(self, W_range=0.08):
        """
        initialize all parameters to uniform in +-W_range
        :param W_range:
        :return:
        """
        all_params = L.layers.get_all_params(self.l_out_flat)
        w_init = L.init.Uniform(range=W_range)
        for param in all_params:
            w = param.get_value()
            w[:] = w_init.sample(w.shape)
        return all_params

    @args_from_opt(1)
    def cost_metric(self, flattened_output, features):
        """
        Cross entropy cost metric
        :param flattened_output:
        :param features:
        :return:
        """
        return T.nnet.categorical_crossentropy(flattened_output,
                                               self.target_values.reshape2xy_batch_chunk(
                                                       (-1, features))).mean()


class LearningRateMixin(LasagneBase):
    @property
    def f_train_alpha(self, decay=.95):
        """
        Compiled theano training function that also takes a learning rate parameter
        as well as x and y (alpha, x, y)
        :param decay:
        :return:
        """
        alphaVar = T.fscalar()
        updates = L.updates.rmsprop(self.cost, self.all_params, alphaVar, decay)
        return theano.function(
                [alphaVar, self.l_in.input_var, self.target_values], self.cost,
                updates=updates, allow_input_downcast=True)

    @lru_cache()
    @args_from_opt(2)
    def f_train(self, alpha=None, step=None, start_alpha=(2 * 10 ** (-3)), alpha_factor=.95):
        """
        produce wrapped f_train_alpha.
        wrapped f_train can be called without alpha parameter
        :param alpha: the learning rate to use. None -> default
        :param step: calculate learningrate from  start_alpha * alpha_factor ** step
        :return: partial function
        """
        f = self.f_train_alpha
        if alpha is None:
            alpha = start_alpha * alpha_factor ** step

        return partial(f, alpha)


class DropoutInOutMixin(LasagneBase):
    @args_from_opt(1)
    def apply_dropout(self, l_prev, dropout=.5):
        return L.layers.DropoutLayer(l_prev, p=dropout)

    @property
    def l_bottom(self):
        return self.apply_dropout(super().l_bottom)

    @property
    def l_top(self):
        return self.apply_dropout(super().l_top)