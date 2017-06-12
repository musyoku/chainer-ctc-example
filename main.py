from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, math, cupy
import numpy as np
from chainer import optimizers, cuda
from chainer import functions as F
from chainer.links import EmbedID, ConvolutionND
from qrnn import QRNN

def compute_character_error_rate(r, h):
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
	for i in xrange(len(r) + 1):
		for j in xrange(len(h) + 1):
			if i == 0: d[0][j] = j
			elif j == 0: d[i][0] = i
	for i in xrange(1, len(r) + 1):
		for j in xrange(1, len(h) + 1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitute = d[i-1][j-1] + 1
				insert = d[i][j-1] + 1
				delete = d[i-1][j] + 1
				d[i][j] = min(substitute, insert, delete)
	return float(d[len(r)][len(h)]) / len(r)

class Model(chainer.Chain):
	def __init__(self, vocab_size, ndim_embedding, num_layers, ndim_h, kernel_size=4, pooling="fo", zoneout=0, dropout=0, wgain=1, densely_connected=False):
		super(Model, self).__init__(
			embed=EmbedID(vocab_size, ndim_embedding),
			dense=ConvolutionND(1, ndim_h, vocab_size, ksize=1, stride=1, pad=0)
		)
		assert num_layers > 0
		self.vocab_size = vocab_size
		self.ndim_embedding = ndim_embedding
		self.num_layers = num_layers
		self.ndim_h = ndim_h
		self.kernel_size = kernel_size
		self.using_dropout = True if dropout > 0 else False
		self.dropout = dropout
		self.densely_connected = densely_connected

		self.add_link("qrnn0", QRNN(ndim_embedding, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wgain=wgain))
		for i in xrange(1, num_layers):
			self.add_link("qrnn{}".format(i), QRNN(ndim_h, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wgain=wgain))

	def get_rnn_layer(self, index):
		return getattr(self, "qrnn{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_layers):
			self.get_rnn_layer(i).reset_state()

	def _forward_layer(self, layer_index, in_data):
		if self.dropout:
			in_data = F.dropout(in_data, ratio=self.dropout)
		rnn = self.get_rnn_layer(layer_index)
		out_data = rnn(in_data)
		return out_data

	def __call__(self, X, split_into_variables=True):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)
		in_data = []
		if self.ndim_embedding == self.ndim_h:
			in_data.append(enmbedding)

		out_data = self._forward_layer(0, enmbedding)
		in_data.append(out_data)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_layer(layer_index, sum(in_data) if self.densely_connected else in_data[-1])	# dense conv
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv

		if self.dropout:
			out_data = F.dropout(out_data, ratio=self.dropout)

		out_data = self.dense(out_data)

		if split_into_variables:
			out_data = F.swapaxes(out_data, 1, 2)
			out_data = F.reshape(out_data, (batchsize, -1))
			out_data = F.split_axis(out_data, seq_length, axis=1)
		else:
			out_data = F.swapaxes(out_data, 1, 2)

		return out_data

def generate_data():
	x_batch = np.zeros((args.dataset_size, args.sequence_length), dtype=np.int32)
	t_batch = np.zeros((args.dataset_size, args.true_sequence_length), dtype=np.int32)
	for data_idx in xrange(len(x_batch)):
		indices = np.random.choice(np.arange(args.sequence_length), size=args.true_sequence_length)
		tokens = np.random.choice(np.arange(1, args.vocab_size), size=args.true_sequence_length)
		for token_idx, (t, token) in enumerate(zip(indices, tokens)): 
			x_batch[data_idx, t] = token
			t_batch[data_idx, token_idx] = token
	return x_batch, t_batch

def main():
	model = Model(args.vocab_size, args.ndim_embedding, args.num_layers, args.ndim_h)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	train_data, train_labels = generate_data()
	total_loop = int(math.ceil(len(train_data) / args.batchsize))
	train_indices = np.arange(len(train_data), dtype=int)

	xp = model.xp
	x_batch_length = xp.full((args.batchsize,), args.sequence_length, dtype=xp.int32)
	t_batch_length = xp.full((args.batchsize,), args.true_sequence_length, dtype=xp.int32)

	# optimizer
	optimizer = optimizers.Adam(args.learning_rate, 0.9)
	optimizer.setup(model)
	

	for epoch in xrange(1, args.total_epoch + 1):
		# train loop
		with chainer.using_config("train", True):
			for itr in xrange(1, total_loop + 1):
				# sample minibatch
				np.random.shuffle(train_indices)
				x_batch = train_data[train_indices[:args.batchsize]]
				t_batch = train_labels[train_indices[:args.batchsize]]

				# GPU
				if xp is cupy:
					x_batch = cuda.to_gpu(x_batch.astype(xp.int32))
					t_batch = cuda.to_gpu(t_batch.astype(xp.int32))

				# forward
				model.reset_state()
				y_batch = model(x_batch)	# list of variables

				# compute loss
				loss = F.connectionist_temporal_classification(y_batch, t_batch, 0, x_batch_length, t_batch_length)
				optimizer.update(lossfun=lambda: loss)

		# evaluate
		with chainer.using_config("train", True):
			# sample minibatch
			np.random.shuffle(train_indices)
			x_batch = train_data[train_indices[:args.batchsize]]
			t_batch = train_labels[train_indices[:args.batchsize]]

			# GPU
			if xp is cupy:
				x_batch = cuda.to_gpu(x_batch.astype(xp.int32))
				t_batch = cuda.to_gpu(t_batch.astype(xp.int32))

			# forward
			model.reset_state()
			y_batch = model(x_batch, split_into_variables=False)
			y_batch = xp.argmax(y_batch.data, axis=2)

			average_error = 0
			for argmax_tokens, true_sequence in zip(y_batch, t_batch):
				pred_seqence = []
				for token in argmax_tokens:
					if token == 0:
						continue
					pred_seqence.append(int(token))
				print(true_sequence, pred_seqence)
				error = compute_character_error_rate(true_sequence.tolist(), pred_seqence)
				average_error += error
			print("CER: {}".format(int(average_error / args.batchsize * 100)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-epoch", "-epoch", type=int, default=100)
	parser.add_argument("--batchsize", "-b", type=int, default=32)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
	parser.add_argument("--vocab-size", "-vocab", type=int, default=50)
	parser.add_argument("--num-layers", "-layers", type=int, default=1)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=50)
	parser.add_argument("--ndim-h", "-nh", type=int, default=128)
	parser.add_argument("--true-sequence-length", "-tseq", type=int, default=5)
	parser.add_argument("--sequence-length", "-seq", type=int, default=30)
	parser.add_argument("--dataset-size", "-size", type=int, default=500)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	args = parser.parse_args()
	main()