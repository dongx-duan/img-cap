import json
import scipy.io
import numpy
import random
import time

from lstm import *
from metrics import Metrics
from collections import Counter

class Dataset:
	def load_file(self, textfile="flickr8k/dataset.json",
			imagefeature="flickr8k/vgg_feats.mat" ):
		
		image = scipy.io.loadmat(imagefeature)
		image = image['feats']
		self.img_emb_size = len(image)
		
		txt = json.load(open(textfile, 'r'))

		self.train = []
		self.cv = []
		self.test = []
		wordCounter = Counter()

		# create word dict
		for single_img in txt['images']:
			for sentence in single_img['sentences']:
				wordCounter.update( Counter(sentence['tokens']) )
				

		self.d = dict()
		for i, k in enumerate(sorted([word for word, count in wordCounter.viewitems() if count >= 5] )):
			self.d[k] = i+1

		self.vocabulary_size = len(self.d) + 2

		self.rd = dict([(v, k) for k,v in self.d.items()])

		start_tag = '__$'
		end_tag = '$__'

		self.start_word_id = 0
		self.end_word_id = self.vocabulary_size - 1

		self.rd[ self.start_word_id ] = start_tag
		self.rd[ self.end_word_id] = end_tag

		# split word sets
		for single_img in txt['images']:
			# print "loading: " + str(sentence['imgid'])	
			tmp = {}
			tmp['fl'] = single_img['filename']
			tmp['id'] = single_img['imgid']
			tmp['xi'] = image[:, single_img['imgid']]
			tmp['ref'] = [" ".join( filter(lambda x: x in self.d, s['tokens']) ) for s in single_img['sentences'] ]
			tmp['tokens'] = []
			for sentence in single_img['sentences']:
				tokens = [ self.d[t] for t in sentence['tokens'] if t in self.d ]
				tmp['tokens'].append( tokens )
			
			if single_img['split'] == 'train':
				self.train.append( tmp )
			elif single_img['split'] == 'test':
				self.test.append( tmp )
			else:
				self.cv.append( tmp )

		
		self.train_size = len(self.train)
		self.cv_size = len(self.cv)
		self.test_size = len(self.test)

		

	def random_training_sample(self, num=1):
		return random.sample(self.train, num)

	def train_sample(self):
		random.shuffle(self.train)
		return self.train

	def cv_sample(self):
		return self.cv

	def test_sample(self):
		return self.test

	def statistic(self):
		print "vocabulary size: " + str(self.vocabulary_size)
		print "train set: " + str(self.train_size)
		print "cv set: " + str(self.cv_size)
		print "test set: " + str(self.test_size)



def train(tid, dataset, model, 
		epoch = 1000, 
		cv_step = 1,
		lr_decay_step = 300,
		lr = 0.01, 
		drop_input=0.2,
		drop_output=0.5,
		savemodel=True):
	
	def compute_cost(f, samples):
		s = 0
		sentence_count = 0
		for rd in samples:
			xi = rd['xi']
			for tokens in rd['tokens'] :
				xsi = generate_one_hot_matrix(len(tokens)+1, dataset.vocabulary_size, 
							[(r, c) for r,c in enumerate([dataset.start_word_id] + tokens)])

				xso = generate_one_hot_matrix(len(tokens)+1, dataset.vocabulary_size, 
							[(r, c) for r,c in enumerate(tokens+[dataset.end_word_id])])

				cost, _, _ = f(xi, xsi, xso, lr, drop_input, drop_output)
				s += cost
				sentence_count += 1
		return s / sentence_count

	min_cv_cost = 2.8 # a hack for save model, otherwise all cv_test will be saved.
	cv_iter = 0
	cv_cost_record = []
	cv_test_record = []
	print "start training " + time.strftime("%Y-%m-%d %H:%M:%S")
	for i in range(epoch):
		cost_train = compute_cost(model.forward, dataset.train_sample())
		print "epoch " + str(i) + "  " + time.strftime("%Y-%m-%d %H:%M:%S") + " --> " + str(cost_train)
		
		# cv 
		if i == cv_iter:
			cv_iter += cv_step

			cost_cv = compute_cost(model.forward_without_update, dataset.cv_sample() )
			print "cost_cv  " + str( cost_cv )
			cost_test = compute_cost(model.forward_without_update, dataset.test_sample() )
			print "cost_test  " + str( cost_test )
			cv_cost_record.append(cost_cv)
			cv_test_record.append(cost_test)
			
			if cost_cv < min_cv_cost:
				min_cv_cost = cost_cv
				if savemodel:
					model.savemodel(tid + time.strftime("%Y_%m_%d_%H_%M_") +str(i) + "_lr_" + str(lr) +
						  "_cv_{0:.4f}".format(cost_cv) +
						  "_test_{0:.4f}".format(cost_test) +".pkl" )
			elif len(cv_cost_record) > 1 and cost_cv > cv_cost_record[-2]:
				print "cost_cv increase. early stop."
				break
		if (i+1) % lr_decay_step == 0:
			lr = lr * 0.1
			print "change learning rate:" + str(lr)

		print ""

def compute_bleu(dataset, model):
	print ""
	print "Test Bleu Score >>>>> "
	metrics = Metrics()
	hypo, ref = model.generate_eval_data(dataset.test_sample(), 
					dataset.rd,
					dataset.start_word_id, 
					dataset.end_word_id, 
					dataset.vocabulary_size)
	
	for k, h, r in zip(hypo.keys(), hypo.values(), ref.values()):
		# print k
		print h
		print r
		print ""
		break

	scores = metrics.bleu( hypo, ref )
	print(scores)