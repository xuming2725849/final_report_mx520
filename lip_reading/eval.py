import string
import pandas as pd

from builtins import dict
from nltk import sent_tokenize, word_tokenize

from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice


class EvalCap:
    def __init__(self, true_list, pred_list):

#         self.pred_df = pd.read_csv(pred_df_csv, header=None).values
#         self.true_df = pd.read_csv(true_df_csv, header=None).values
        self.true = true_list
        self.pred = pred_list

        self.eval = dict()
        self.imgToEval = dict()

    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        # s = s.translate(str.maketrans('', '', '0123456789'))
        # s = s.translate(str.maketrans('', '', string.punctuation))
        return s

    def evaluate(self):

        gts = dict()
        res = dict()

        # Sanity Checks
        assert len(self.pred) == len(self.true)

        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for i in range(len(self.pred)):
#             pred_text = ' '.join(word_tokenize(self.preprocess(self.pred_df[i][0])))
#             true_text = ' '.join(word_tokenize(self.preprocess(self.true_df[i][0])))
            pred_text = ' '.join(word_tokenize(self.preprocess(self.pred[i])))
            true_text = ' '.join(word_tokenize(self.preprocess(self.true[i])))
            res[i] = [pred_text]
            gts[i] = [true_text]

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
#            (Meteor(),"METEOR"),
#             (Rouge(), "ROUGE_L"),
#             (Cider(), "CIDEr"),
#             (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
