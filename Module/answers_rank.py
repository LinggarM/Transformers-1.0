import tensorflow as tf
import numpy as np

class AnswerRank :

    def __init__(self, embedded_pertanyaan, embedded_jawaban, njawaban) :
        self.embedded_pertanyaan = embedded_pertanyaan
        self.embedded_jawaban = embedded_jawaban
        self.njawaban = njawaban
        
        # measure the distance
        self.measureDistance()

        # sort jawaban
        self.idx_sorted = np.argsort(self.dist_list)

    def measureDistance(self) :
        dist_list = []
        for i in range(self.njawaban) :
            dist = cosineSimilarity(self.embedded_pertanyaan[0], self.embedded_jawaban[i][0])
            dist_list.append(float(dist))
        self.dist_list = dist_list
    
    def avgPooling(self, x) :
        return K.mean(K.constant(x), axis=1, keepdims=False)
    
    def cosineSimilarity(self, x, y) :
        normalize_a = tf.nn.l2_normalize(avgPooling(x), 0)        
        normalize_b = tf.nn.l2_normalize(avgPooling(y), 0)
        return tf.reduce_sum(tf.multiply(normalize_a,normalize_b))