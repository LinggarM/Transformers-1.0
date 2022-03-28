from data import Data
from word_embedding import BERTEmbedding
from answer_rank import AnswerRank
from selected_answer import SelectedAnswer
from model import CNNLSTMModel
from train_test_data import TrainTestData

# Data
data = Data()
print("DATA")
print("\n", data.pertanyaan)
print("\n", data.id_mahasiswa)
print("\n", data.jawaban_list)
print("\n", data.njawaban)

# Embedding
embedding = BERTEmbedding(
    data.pertanyaan,
    data.jawaban_list,
    data.njawaban)
embedding.encoded_pertanyaan
embedding.encoded_jawaban
embedding.encoding_length
embedding.embedded_pertanyaan
embedding.embedded_jawaban

# Answer Rank
answer_rank = AnswerRank(
    embedding.embedded_pertanyaan,
    embedding.embedded_jawaban,
    data.njawaban)

print("\nANSWER RANK")
print("\n", answer_rank.idx_sorted)
print("\n", answer_rank.dist_list)

# Selected Answer
selected_answer = SelectedAnswer(
    answer_rank.idx_sorted,
    20,
    data.njawaban)

print("\nSELECTED ANSWER")
print("\n", selected_answer.idx_selected)
print("\n", selected_answer.nselected)
print("\n", selected_answer.nselected_each)
print("\n", selected_answer.nselected_rest)

# labels = []
# for i in selected_answer.idx_selected :
#   print(i, data.jawaban_list[i])
#   label = input("Beri skor : ")
#   labels.append(label)
# dummy labels
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
selected_answer.setLabels(labels)

print("\n", selected_answer.labels)
print("\n", sorted(set(selected_answer.labels)))

# Train Test Data
train_test_data = TrainTestData(embedding, answer_rank.idx_sorted, selected_answer)

print("\nTRAIN TEST DATA")
print(train_test_data.x_train_pertanyaan.shape)
print(train_test_data.x_train_jawaban.shape)
print(train_test_data.x_test_pertanyaan.shape)
print(train_test_data.x_test_jawaban.shape)
print("\n", train_test_data.y_train)

# Model
model = CNNLSTMModel(embedding, 64, 3, 128, 0.4, 0.4, 0.4, 'mean_squared_error', 'adam', ['accuracy', 'mae'])

model.visualizeModel()
history = model.trainModel(train_test_data.x_train_pertanyaan, train_test_data.x_train_jawaban, train_test_data.y_train, 70)
model.getLossVisualization()