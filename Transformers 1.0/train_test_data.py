import numpy as np

class TrainTestData :

  def __init__(self, embedding, idx_sorted, selected_answer) :

    # train data
    self.x_train_pertanyaan = np.array([embedding.embedded_pertanyaan for i in range(selected_answer.nselected)])
    self.x_train_jawaban = np.array([embedding.embedded_jawaban[i] for i in selected_answer.idx_selected])
    self.y_train = np.array(selected_answer.labels)

    # index for test data
    self.idx_test = list(set(idx_sorted) - set(selected_answer.idx_selected))

    # test data
    self.x_test_pertanyaan = np.array([embedding.embedded_pertanyaan for i in range(len(self.idx_test))])
    self.x_test_jawaban = np.array([embedding.embedded_jawaban[i] for i in idx_test])