from __future__ import division
from __future__ import print_function
from six.moves import xrange
import math
import numpy as np
import chainer
from chainer import cuda, Variable, function, link, functions, links, initializers
from chainer.utils import type_check
from chainer.links import EmbedID, Linear, BatchNormalization, ConvolutionND

class Zoneout(function.Function):
	def __init__(self, p):
		self.p = p

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		type_check.expect(in_types[0].dtype.kind == 'f')

	def forward(self, x):
		if not hasattr(self, "mask"):
			xp = cuda.get_array_module(*x)
			if xp == np:
				flag = xp.random.rand(*x[0].shape) >= self.p
			else:
				flag = xp.random.rand(*x[0].shape, dtype=np.float32) >= self.p
			self.mask = flag
		return x[0] * self.mask,

	def backward(self, x, gy):
		return gy[0] * self.mask,

def zoneout(x, ratio=.5):
	return Zoneout(ratio)(x)

class QRNN(link.Chain):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=0, wgain=1):
		self.num_split = len(pooling) + 1
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		super(QRNN, self).__init__(W=ConvolutionND(1, in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1, initialW=initializers.Normal(wstd)))
		self._in_channels, self._out_channels, self._kernel_size, self._pooling, self._zoneout = in_channels, out_channels, kernel_size, pooling, zoneout
		self._using_zoneout = True if self._zoneout > 0 else False
		self.reset_state()

	def __call__(self, X, skip_mask=None):
		pad = self._kernel_size - 1
		WX = self.W(X)[..., :-pad]

		return self.pool(functions.split_axis(WX, self.num_split, axis=1), skip_mask=skip_mask)

	def forward_one_step(self, X, skip_mask=None):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		return self.pool(functions.split_axis(WX, self.num_split, axis=1), skip_mask=skip_mask)

	def zoneout(self, U):
		if self._using_zoneout and chainer.config.train:
			return 1 - zoneout(functions.sigmoid(-U), self._zoneout)
		return functions.sigmoid(U)

	def pool(self, WX, skip_mask=None):
		Z, F, O, I = None, None, None, None

		# f-pooling
		if len(self._pooling) == 1:
			assert len(WX) == 2
			Z, F = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)

		# fo-pooling
		if len(self._pooling) == 2:
			assert len(WX) == 3
			Z, F, O = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)
			O = functions.sigmoid(O)

		# ifo-pooling
		if len(self._pooling) == 3:
			assert len(WX) == 4
			Z, F, O, I = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)
			O = functions.sigmoid(O)
			I = functions.sigmoid(I)

		assert Z is not None
		assert F is not None

		T = Z.shape[2]
		for t in xrange(T):
			zt = Z[..., t]
			ft = F[..., t]
			ot = 1 if O is None else O[..., t]
			it = 1 - ft if I is None else I[..., t]
			xt = 1 if skip_mask is None else skip_mask[:, t, None]	# will be used for seq2seq to skip PAD

			if self.ct is None:
				self.ct = (1 - ft) * zt * xt
			else:
				self.ct = ft * self.ct + it * zt * xt
			self.ht = self.ct if O is None else ot * self.ct

			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H

	def reset_state(self):
		self.set_state(None, None, None)

	def set_state(self, ct, ht, H):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.H = H		# all hidden states

	def get_last_hidden_state(self):
		return self.ht

	def get_all_hidden_states(self):
		return self.H