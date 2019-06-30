import os
import gensim
from gensim.models import word2vec
from sklearn.decomposition import PCA
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

class WordVector:
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format("model/token_vec_300.bin", binary=False)
    def getWordVector(self,string):
        res = np.zeros((300))
        for char in string:
            try:
                wordV = self.model.get_vector(char)
                res += wordV
            except:
                pass
        res /= len(string)
        return res

if __name__ == '__main__':
    w = WordVector()
    print(w.getWordVector("上海财经大学"))