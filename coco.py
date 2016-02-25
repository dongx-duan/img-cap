import sys

from dataset import *
from lstm import *


if __name__ == '__main__':
	
	print time.strftime("%Y-%m-%d %H:%M:%S")

	dataset = Dataset()
	dataset.load_file(textfile="coco/dataset.json",	imagefeature="coco/vgg_feats.mat")
	dataset.statistic()
	

	mode = 'training'
	# mode = 'predict'
	filename = None
	if len( sys.argv ) == 3:
		mode = sys.argv[1]
		filename = sys.argv[2]

	model = None

	if mode == 'training':
		model = LSTM(dataset.img_emb_size, dataset.vocabulary_size, 512, 512, filename)
		train('flickr8k', dataset, model, 100, lr_decay_step = 40, lr =  0.01)
	elif mode == 'eval':
		model = LSTM(dataset.img_emb_size, dataset.vocabulary_size, 512, 512, filename)
	else:
		print 'unknown mode, should be training or predict'
		exit(0)

	print "Sample 5 Predicts >>>>>> "
	for r in dataset.random_training_sample(5):
		#predict(self, Xi, start_word_id, end_word_id, vcab_size, topk, max_length):
		candidates = model.predict( r['xi'], dataset.start_word_id, dataset.end_word_id, dataset.vocabulary_size, 10, 20)

		print r['fl'] 
		#print_predict(candidates, word_dict, start_word_id, end_word_id)
		model.print_predict( candidates, dataset.rd, dataset.start_word_id, dataset.end_word_id)
		print "\n\n"
	compute_bleu(dataset, model)
	