from data import Data
from word_embedding import BERTEmbedding
from answers_rank import AnswerRank

data = Data()
print(data.pertanyaan)
print(data.id_mahasiswa)
print(data.jawaban_list)
print(data.njawaban)

embedding = BERTEmbedding(
    data.pertanyaan,
    data.jawaban_list,
    data.njawaban)
embedding.encoded_pertanyaan
embedding.encoded_jawaban
embedding.encoding_length
embedding.embedded_pertanyaan
embedding.embedded_jawaban

answer_rank = AnswerRank(
    embedding.embedded_pertanyaan,
    embedding.embedded_jawaban,
    data.njawaban)

answer_rank.idx_sorted
answer_rank.dist_list

print(answer_rank.dist_list)