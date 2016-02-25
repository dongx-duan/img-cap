from __future__ import absolute_import

import sys
sys.path.append("/home/hadoop/ddx/lstm/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

class Metrics:
	def __init__(self):
		pass

	def bleu(self, hypo, ref):
		self.bleu_scorer = Bleu(4)
		final_scores = {}
		score, scores = self.bleu_scorer.compute_score(ref, hypo)
		for m, s in zip(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], score):
			final_scores[m] = s
		return final_scores