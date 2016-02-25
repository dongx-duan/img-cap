from collections import OrderedDict

import theano
import numpy
import copy
from theano import tensor

import cPickle as pickle


def generate_one_hot(fullsize, i):
	v = numpy.zeros(fullsize).astype(numpy.bool)
	v[i] = 1.
	return v

def generate_one_hot_matrix(row, col, set_one):
	v = numpy.zeros( (row, col) ).astype(numpy.bool)
	for r, c in set_one:
		v[r][c] = 1
	return v

def sigmoid(x):
	return 1 / (1 + numpy.exp(-x))

def softmax(w, t=1.0):
	e = numpy.exp(w / t)
	return e / numpy.sum(e)


theano.config.floatX = "float32"


class LSTM:
	# ni: length of image vector
	# ns: length of one hot word vector
	# ne: length of embedding vector
	# nh: length of hidden state vector
	# lr: learning rate
	# reload: file of stored params
	def __init__(self, ni, ns, ne, nh, reload = None):

		self.ni = ni
		self.ns = ns
		self.ne = ne
		self.nh = nh

		self.momentum = 0.9

		if reload:
			self.wxi, self.wxs, self.W, self.U, self.B = self.loadmodel(reload)
		else:
			# convert image vector to hidden vector
			self.wxi = theano.shared(name="image_embedding", 
						value=numpy.random.uniform(-0.1, 0.1, (ne, ni)).astype(theano.config.floatX))
			self.wxs = theano.shared(name="word_embedding",
						value=numpy.random.uniform(-0.1, 0.1, (ns, ne)).astype(theano.config.floatX))
			# lstm weights
			# W[:nh]        for input gate
			# W[nh+1: 2*nh]  for forget gate
			# W[2nh+1, 3*nh] for out gate
			# W[3*nh + 1:]  for 
			self.W = theano.shared(name="lstm",
						value=numpy.random.uniform(-0.1, 0.1, (ne + nh + 1, 4 * nh)).astype(theano.config.floatX))
			
			# for softmax p = softmax(m)
			self.U = theano.shared(name="softmax_w",
						value=numpy.random.uniform(-0.1, 0.1, (ns, nh)).astype(theano.config.floatX))
			self.B = theano.shared(name="softmax_b",
						value=numpy.zeros(ns).astype(theano.config.floatX))

		# update hidden state m, c for each time t
		dd = 1 + ne + nh
		def _step(_s, _din, _dout, _m, _c, _p):
			hin = tensor.concatenate([numpy.asarray([1.]).astype(theano.config.floatX), _s * _din, _m])
			#hin = numpy.concatenate( [numpy.array([1.]).astype(theano.config.floatX), _s, _m])
			ifog = tensor.dot(hin, self.W)
			ifo = tensor.nnet.sigmoid( ifog[:3*nh] )
			g = tensor.tanh(ifog[ 3*nh: ])

			_c = ifo[nh : 2*nh] * _c + ifo[:nh] * g
			_m = ifo[2*nh: 3*nh] * _c
			# apply drop out on non-recurrent connections
			_drop_out_m = _m * _dout
			_p = tensor.nnet.softmax(tensor.dot(self.U, _drop_out_m) + self.B).flatten()
			return _m, _c, _p

		Xi = tensor.vector()
		Xi_emb = tensor.dot(self.wxi, Xi).flatten()
		

		m0, c0, _= _step(Xi_emb, 
				numpy.ones(ne).astype(theano.config.floatX),
				numpy.ones(nh).astype(theano.config.floatX),
				numpy.zeros(nh).astype(theano.config.floatX),
				numpy.zeros(nh).astype(theano.config.floatX),
				numpy.zeros(ns).astype(theano.config.floatX))

		# words input sequence!!!!
		Xsi = tensor.fmatrix()
		X_emb = tensor.dot(Xsi, self.wxs)

		# word output sequence!!!!
		Xso = tensor.fmatrix()

		# input drop out
		D_word = tensor.fmatrix()
		# output drop out
		D_out = tensor.fmatrix()


		rev, updates = theano.scan(_step, 
			sequences=[X_emb, D_word, D_out],
			outputs_info=[m0, c0, tensor.alloc(numpy.asarray(0., dtype=theano.config.floatX), ns)]
		)
		
		# cost function and predict word id for debug.
		masked_p =  rev[2] * Xso
		predict_p = tensor.max( masked_p, axis=1 )
		predict_id = tensor.argmax( masked_p, axis =1 ) 

		cost = - tensor.log( predict_p + 1e-7).mean()

		self.params = [self.wxi, self.wxs, self.W, self.U, self.B]
		self.grad = tensor.grad(cost, self.params)
		self._cache_grad = [theano.shared(numpy.zeros_like(_p.get_value())) for _p in self.params ]

		lr = tensor.fscalar('lr')
		
		
		self.updates = OrderedDict( (_m, self.momentum  *_m + lr * _g) for _m, _g in zip(self._cache_grad, self.grad) )
		self.updates.update( (_p, _p - _m) for _p, _m in zip(self.params, self._cache_grad) )

		self._forward = theano.function(inputs=[Xi, Xsi, Xso, D_word, D_out, lr], 
				outputs=[cost, predict_p, predict_id], 
				updates=self.updates)
		
		self._forward_without_update = theano.function(inputs=[Xi, Xsi, Xso, D_word, D_out, lr], 
					outputs=[cost, predict_p, predict_id], 
					on_unused_input='ignore' )
		

	def drop_out(self, X, drop_rate):
		D = None
		if drop_rate:
			D = (numpy.random.rand(X.shape[0], self.nh) < (1 - drop_rate) ) * (1.0 / (1.0 - drop_rate))
		else:
			D = numpy.ones((X.shape[0], self.nh))
		return D.astype(theano.config.floatX)

	def drop_out_input(self, X, drop_rate):
		if drop_rate:
			D = (numpy.random.rand( X.shape[0], self.ne ) < (1 - drop_rate) ) * (1.0 / (1.0 - drop_rate))
			return D.astype(theano.config.floatX)
		else:
			return numpy.ones((X.shape[0], self.ne)).astype(theano.config.floatX)

	def forward(self, Xi, Xsi, Xso, lr, drop_input, drop_output):
		D1 = self.drop_out_input(Xsi, drop_input)
		D2 = self.drop_out(Xsi, drop_output)
		return self._forward(Xi, Xsi, Xso, D1, D2, lr)

	def forward_without_update(self, Xi, Xsi, Xso, lr, drop_input, drop_output):
		D1 = self.drop_out_input(Xsi, None)
		D2 = self.drop_out(Xsi, None)
		return self._forward_without_update(Xi, Xsi, Xso, D1, D2, lr)

	# Given a vector of image, return topk of the max probabilistic sentences
	# Each sentence has no more than max_length words
	def predict(self, Xi, start_word_id, end_word_id, vcab_size, topk=20, max_length=30):

		def _onestep(s, _m, _c, p):

			hin = numpy.concatenate( [numpy.array([1.]).astype(numpy.float32), s, _m])
			ifog = numpy.dot(hin, self.W.get_value())
			ifo = sigmoid( ifog[:3* self.nh] )
			g =  numpy.tanh(ifog[ 3* self.nh: ])

			_c = ifo[self.nh : 2*self.nh] * _c + ifo[:self.nh] * g
			_m = ifo[2*self.nh: 3*self.nh] * _c
			_p = softmax(numpy.dot(self.U.get_value(), _m) + self.B.get_value()).flatten()
			return _m, _c, _p

		m, c, _ = _onestep( numpy.dot(self.wxi.get_value(), Xi).flatten(), 
				numpy.zeros(self.nh).astype(numpy.float32),
				numpy.zeros(self.nh).astype(numpy.float32),
				numpy.zeros(self.ns).astype(numpy.float32) )

		#(log probability,  m,  c,  p,  [list of word_ids])
		candidates = [([], 
			m, 
			c, 
			numpy.zeros(self.ns).astype(numpy.float32), 
			[start_word_id], [1])]


		for i in range(0, max_length):
			current = []
			for tmp in candidates:
				#print tmp[4]
				word = tmp[4][-1]
				# end of sentence
				if word != end_word_id:
					Xs = generate_one_hot(vcab_size, word)
					_m, _c, _p = _onestep( numpy.dot(Xs, self.wxs.get_value()), 
						copy.deepcopy(tmp[1]), 
						copy.deepcopy(tmp[2]), 
						copy.deepcopy(tmp[3]) )

					wids = numpy.argsort(_p)[-topk:]
					wps = numpy.sort(_p)[-topk:]
					
					for(wid, wp) in zip(wids, wps):
						_tmp = copy.deepcopy(tmp[0])
						if wid != end_word_id:
							_tmp = _tmp + [numpy.log(wp + 1e-7)] 
						current.append((_tmp, _m, _c, _p, 
							copy.deepcopy(tmp[4]+[wid]), 
							copy.deepcopy(tmp[5]+[wp] ) ))
				else:
					current.append(tmp)

			current = sorted(current, key=lambda t: numpy.mean(t[0]), reverse=True)

			candidates = current[:min(len(current), topk)]
			

		return candidates


	# Print the predict sentences of an image
	def print_predict(self, candidates, word_dict, start_word_id, end_word_id):
		for c in candidates[:5]:
			print numpy.exp( numpy.mean( c[0] ) )
			print "    ".join([str(word_dict[i]) + "|" + "{:.2f}".format(p) for i, p in zip(c[4], c[5])
				 				if i != start_word_id and i != end_word_id ])
			print ""
		print ""

	# For bleu, only return the first candidate sentence
	def generate_sentence(self, candidates, word_dict, sid, eid):
		return [ " ".join( str(word_dict[i]) for i in c[4] if i != sid and i != eid ) for c in candidates[:1] ]

	def generate_eval_data(self, images, word_dict, start_word_id, end_word_id, vcab_size):
		hypo = dict( [(img['id'], 
				self.generate_sentence( self.predict( img['xi'], start_word_id, end_word_id, vcab_size), 
					word_dict, start_word_id, end_word_id)) for img in images] )

		ref = dict( [(img['id'], img['ref']) for img in images] )
		return hypo, ref

	def savemodel(self, fname):
		pickle.dump( self.params, open(fname, 'wb') )


	def loadmodel(self, fname):
		return pickle.load(open(fname, 'rb') )


