"""
Instructions on how to put together different kinds of lasagnas
"""
# builtins
from collections import (namedtuple, defaultdict, OrderedDict)
from functools import (lru_cache, partial)

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
from .fridge import ClassSaveLoadMixin


class LasagneBase(ChainedProps, ClassSaveLoadMixin,
                  metaclass=ChainPropsABCMetaclass):
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

    @property
    def saved_params(self):
        """
        A property to hold saved parameters that should be used to initialize
        :return: None by default. But will can be set by a cook at .open_shop()
        if any aparams exists in in the recipe tupperware
        """
        return None

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

    def compiled_function(self, *args, **kwargs):
        return theano.function(*args, **kwargs)

    @property
    def f_train(self):
        """
        Compiled theano function that takes (x, y) and trains the lasagne
        Updates use default cost (e.g with dropout)
        But f_train return the deterministic cost
        :return:
        """
        updates = L.updates.rmsprop(self.cost, self.all_train_params, .002)
        return self.compiled_function([self.l_in.input_var, self.target_values],
                               self.cost_det,
                               updates=updates, allow_input_downcast=True)

    @property
    def f_train_noreturn(self):
        """
        Compiled theano function that takes (x, y) and trains the lasagne
        Updates use default cost (e.g with dropout)
        Does *not* return a cost.
        :return: None
        """
        updates = L.updates.rmsprop(self.cost, self.all_train_params, .002)
        return self.compiled_function([self.l_in.input_var, self.target_values],
                               updates=updates, allow_input_downcast=True)

    @property
    def f_cost(self):
        """
        Compiled theano function that takes (x, y) and return cost.
        No updates is made
        :return:
        """
        return self.compiled_function([self.l_in.input_var, self.target_values],
                               self.cost_det, allow_input_downcast=True)

    @property
    def f_predict(self):
        """
        Compiled theano function that takes (x) and predicts y
        Computed *deterministic*
        :return:
        """
        output = L.layers.get_output(self.l_out, deterministic=True)
        return self.compiled_function([self.l_in.input_var], output,
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
    def make_recurrent_layer(self, l_prev, n_hid_unit, grad_clip=100):
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


class LSTMStateReuse(LSTMBase):
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
                return x[:, n*self.num_units:(n+1)*self.num_units]

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
                    ingate += cell_previous*self.W_cell_to_ingate
                    forgetgate += cell_previous*self.W_cell_to_forgetgate

                # Apply nonlinearities
                ingate = self.nonlinearity_ingate(ingate)
                forgetgate = self.nonlinearity_forgetgate(forgetgate)
                cell_input = self.nonlinearity_cell(cell_input)

                # Compute new cell value
                cell = forgetgate*cell_previous + ingate*cell_input

                if self.peepholes:
                    outgate += cell*self.W_cell_to_outgate
                outgate = self.nonlinearity_outgate(outgate)

                # Compute new hidden unit activation
                hid = outgate*self.nonlinearity(cell)
                return [cell, hid]

            def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
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
            return cell_out, hid_out

    class ExtractLSTMOut(L.layers.Layer):
        def __init__(self, input_layer: L.layers.LSTMLayer, name=None):
            super().__init__(input_layer, name=name)
            self.input_layer = input_layer

        def get_output_shape_for(self, input_shape):
            return input_shape

        def get_output_for(self, input, **kwargs):
            cell_out, hid_out = input

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
        def __init__(self, input_layer: L.layers.LSTMLayer, name=None):
            super().__init__(input_layer, name=name)
            self.input_layer = input_layer

        def get_output_for(self, input, **kwargs):
            cell_out, hid_out = input
            if self.input_layer.backwards:
                return hid_out[0], cell_out[0]
            return cell_out[-1], hid_out[-1]

    @property
    def hidden_state_updates(self):
        from weakref import WeakKeyDictionary
        return WeakKeyDictionary()


    @args_from_opt('batched')
    def reset_hidden_states(self, batched=True, batch_sz=None, n_hid_unit=None):
        if batched:
            zeros = np.zeros((batch_sz, n_hid_unit), dtype=np.float32)
        else:
            zeros = np.zeros((1, n_hid_unit), dtype=np.float32)

        for layer, (param, update) in self.hidden_state_updates.items():
            param.set_value(zeros)


    @args_from_opt(1)
    def make_recurrent_layer(self, l_prev, batch_sz, seq_len, n_hid_unit, grad_clip=100,
                             unroll=False):
        state_shape = (batch_sz, n_hid_unit)
        h_param = L.utils.create_param(L.init.Constant(0.0), state_shape,
                                       name="h_state")
        c_param = L.utils.create_param(L.init.Constant(0.0), state_shape,
                                       name="c_state")

        hid = L.layers.InputLayer((batch_sz, n_hid_unit), input_var=h_param)
        cell = L.layers.InputLayer((batch_sz, n_hid_unit), input_var=c_param)

        if unroll:
            sys.setrecursionlimit(int(2000 + batch_sz * seq_len))
        raw_lay = self.LSTMLayer(l_prev, n_hid_unit,
                                 hid_init=hid, cell_init=cell,
                                 grad_clipping=grad_clip,
                                 nonlinearity=L.nonlinearities.tanh,
                                 name='LSTM_raw',
                                 unroll_scan=unroll)
        return_lay = self.ExtractLSTMOut(raw_lay, name='LSTM_return')
        state_lay = self.ExtractLastStates(raw_lay, name='LSTM_states')

        cell_upd, hid_upd = L.layers.get_output(state_lay)
        self.hidden_state_updates[hid] = (h_param, hid_upd)
        self.hidden_state_updates[cell] = (c_param, cell_upd)
        return return_lay

    def compiled_function(self, *args, **kwargs):
        updates = kwargs.pop('updates', list())
        if isinstance(updates, dict):
            updates.update(self.hidden_state_updates.values())
        elif isinstance(updates, tuple):
            updates = chain(updates, self.hidden_state_updates.values())
        else:
            updates.extend(self.hidden_state_updates.values())

        return super().compiled_function(*args, updates=updates, **kwargs)


class LearningRateMixin(ChainedProps):
    @property
    def f_train_alpha_noreturn(self, decay=.95):
        """
        Compiled theano training function that also takes a learning rate parameter
        as well as x and y (alpha, x, y)
        :param decay:
        :return:
        """
        alphaVar = T.fscalar()
        updates = L.updates.rmsprop(self.cost, self.all_train_params, alphaVar)
        return self.compiled_function(
            [alphaVar, self.l_in.input_var, self.target_values],
            updates=updates, allow_input_downcast=True)

    @property
    def f_train_alpha(self, decay=.95):
        """
        Compiled theano training function that also takes a learning rate parameter
        as well as x and y (alpha, x, y)
        :param decay:
        :return:
        """
        alphaVar = T.fscalar()
        updates = L.updates.rmsprop(self.cost, self.all_train_params, alphaVar)
        return self.compiled_function(
            [alphaVar, self.l_in.input_var, self.target_values], self.cost_det,
            updates=updates, allow_input_downcast=True)

    @lru_cache()
    @args_from_opt(2)
    def f_train_noreturn(self, alpha=None, step=None,
                         start_alpha=(2 * 10 ** (-3)), alpha_factor=.95):
        """
        produce wrapped f_train_alpha.
        wrapped f_train can be called without alpha parameter
        :param alpha: the learning rate to use. None -> default
        :param step: calculate learningrate from  start_alpha * alpha_factor ** step
        :return: partial function
        """
        f = self.f_train_alpha_noreturn
        if alpha is None:
            alpha = start_alpha * alpha_factor ** step
        print("learning rate: " + str(alpha))
        return partial(f, alpha)

    @lru_cache()
    @args_from_opt(2)
    def f_train(self, alpha=None, step=None,
                         start_alpha=(2 * 10 ** (-3)), alpha_factor=.95):
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
        print("learning rate: " + str(alpha))
        return partial(f, alpha)


class DropoutMixin(ChainedProps):
    @args_from_opt(1)
    def apply_dropout(self, l_prev, specific_dropout=None, dropout=.5):
        if specific_dropout is None:
            return L.layers.DropoutLayer(l_prev, p=dropout)
        return L.layers.DropoutLayer(l_prev, p=specific_dropout)


class DropoutInMixin(DropoutMixin):
    @property
    def l_bottom(self, dropout_in=None):
        return self.apply_dropout(super().l_bottom, specific_dropout=dropout_in)


class DropoutOutMixin(DropoutMixin):
    @property
    def l_top(self, dropout_out=None):
        return self.apply_dropout(super().l_top, specific_dropout=dropout_out)


# noinspection PyUnresolvedReferences
class DropoutBetweenMixin(DropoutMixin):
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


class LSTMDropout(LearningRateMixin, DropoutInMixin, DropoutOutMixin, LSTMBase):
    pass


class LSTMKarpath(LearningRateMixin, DropoutOutMixin, DropoutBetweenMixin,
                  LSTMStateReuse):
    pass


"""
class LSTMDropoutLR(DropoutInOutMixin, LearningRateMixin, LSTMBase):
    @property
    def f_grad(self):
        g = theano.grad(self.cost, self.all_params)
        return theano.function([self.l_in.input_var, self.target_values],
                               [self.cost] + g, allow_input_downcast=True)
"""
