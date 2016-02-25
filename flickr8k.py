import sys

from dataset import *
from lstm import *


if __name__ == '__main__':
	
	print time.strftime("%Y-%m-%d %H:%M:%S")

	dataset = Dataset()
	dataset.load_file(textfile="flickr8k/dataset.json",	imagefeature="flickr8k/vgg_feats.mat")
	dataset.statistic()
	

	mode = 'training'
	# mode = 'predict'
	filename = None
	if len( sys.argv ) == 3:
		mode = sys.argv[1]
		filename = sys.argv[2]

	model = None

	if mode == 'training':
		model = LSTM(4096, dataset.vocabulary_size, 128, 128, filename)
		train('flickr8k_128_', dataset, model, epoch=22, lr_decay_step = 10, lr =  0.01)
	elif mode == 'eval':
		model = LSTM(4096, dataset.vocabulary_size, 512, 512, filename)
	else:
		print 'unknown mode, should be training or predict'
		exit(0)

	print ""
	print "Sample 5 Predicts >>>>>> "
	for r in dataset.random_training_sample(10):
		#predict(self, Xi, start_word_id, end_word_id, vcab_size, topk, max_length):
		candidates = model.predict( r['xi'], dataset.start_word_id, dataset.end_word_id, dataset.vocabulary_size, 10, 20)

		print r['fl'] 
		for ref in r['ref']:
			print ref
		print ""
		#print_predict(candidates, word_dict, start_word_id, end_word_id)
		model.print_predict( candidates, dataset.rd, dataset.start_word_id, dataset.end_word_id)
		print "\n\n"
	compute_bleu(dataset, model)



	


	
