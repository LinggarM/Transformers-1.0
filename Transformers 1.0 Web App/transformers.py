import numpy as np
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import gensim.models
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf
from keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Flatten, Dense, Dropout
from keras.layers.merge import concatenate

class GlobalVariables() :

	def __init__(self) :
		self.id_mahasiswa = []
		self.jawaban = []
		self.idx_labeled = []
		self.idx_sorted = []
		self.pertanyaan_vectors = np.empty([2, 2])
		self.jawaban_results = np.empty([2, 2])

	def set_id_mahasiswa(self, id_mahasiswa) :
		self.id_mahasiswa = id_mahasiswa
		return self.id_mahasiswa

	def set_jawaban(self, jawaban) :
		self.jawaban = jawaban
		return self.jawaban

	def set_idx_labeled(self, idx_labeled) :
		self.idx_labeled = idx_labeled
		return self.idx_labeled

	def set_idx_sorted(self, idx_sorted) :
		self.idx_sorted = idx_sorted
		return self.idx_sorted

	def set_pertanyaan_vectors(self, pertanyaan_vectors) :
		self.pertanyaan_vectors = pertanyaan_vectors
		return self.pertanyaan_vectors

	def set_jawaban_results(self, jawaban_results) :
		self.jawaban_results = jawaban_results
		return self.jawaban_results

class Data :

    def __init__(self) :
        # path
        self.path = "./static/files/data/"
        self.path_pertanyaan = f"{self.path}/pertanyaan.txt"
        self.path_jawaban = f"{self.path}/jawaban.csv"

        # pertanyaan
        self.pertanyaan = open(self.path_pertanyaan).read()

        # jawaban
        self.jawaban_df = pd.read_csv(self.path_jawaban, sep = ';', encoding = "ISO-8859-1")
        self.jawaban = self.jawaban_df['jawaban'].tolist()

        # id_mahasiswa
        self.id_mahasiswa = self.jawaban_df['id_mahasiswa'].tolist()

        # n jawaban
        self.njawaban = len(self.jawaban_df)
    
    def isColumnNormal() :
        return 1

class BERTEmbedding :

	def __init__(self, pertanyaan, jawaban) :
		# Set Pertanyaan & Jawaban
		self.pertanyaan = pertanyaan
		self.jawaban = jawaban

		# Choose BERT model
		self.tfhub_handle_encoder, self.tfhub_handle_preprocess = self.chooseModel('bert_multi_cased_L-12_H-768_A-12')

		# Set preprocess & encoding model
		self.bert_preprocess_model, self.bert_model = self.setModel(self.tfhub_handle_encoder, self.tfhub_handle_preprocess)
		
		# Data Preprocessing
		self.pertanyaan_preprocessed, self.jawaban_preprocessed = self.preprocessData(self.bert_preprocess_model)

		# Word Embedding
		self.pertanyaan_results, self.jawaban_results = self.wordEmbedding(self.bert_model, self.pertanyaan_preprocessed, self.jawaban_preprocessed)

	def chooseModel(self, bert_model_name = 'bert_multi_cased_L-12_H-768_A-12') :
		map_name_to_handle = {
		    'bert_en_uncased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
		    'bert_en_cased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
		    'bert_multi_cased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
		    'small_bert/bert_en_uncased_L-2_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-2_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-2_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-2_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
		    'small_bert/bert_en_uncased_L-4_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-4_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-4_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-4_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
		    'small_bert/bert_en_uncased_L-6_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-6_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-6_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-6_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
		    'small_bert/bert_en_uncased_L-8_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-8_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-8_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-8_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
		    'small_bert/bert_en_uncased_L-10_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-10_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-10_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-10_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
		    'small_bert/bert_en_uncased_L-12_H-128_A-2':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
		    'small_bert/bert_en_uncased_L-12_H-256_A-4':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
		    'small_bert/bert_en_uncased_L-12_H-512_A-8':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
		    'small_bert/bert_en_uncased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
		    'albert_en_base':
		        'https://tfhub.dev/tensorflow/albert_en_base/2',
		    'electra_small':
		        'https://tfhub.dev/google/electra_small/2',
		    'electra_base':
		        'https://tfhub.dev/google/electra_base/2',
		    'experts_pubmed':
		        'https://tfhub.dev/google/experts/bert/pubmed/2',
		    'experts_wiki_books':
		        'https://tfhub.dev/google/experts/bert/wiki_books/2',
		    'talking-heads_base':
		        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
		}

		map_model_to_preprocess = {
		    'bert_en_uncased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'bert_en_cased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
		    'small_bert/bert_en_uncased_L-2_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-2_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-2_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-2_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-4_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-4_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-4_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-4_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-6_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-6_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-6_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-6_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-8_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-8_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-8_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-8_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-10_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-10_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-10_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-10_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-12_H-128_A-2':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-12_H-256_A-4':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-12_H-512_A-8':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'small_bert/bert_en_uncased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'bert_multi_cased_L-12_H-768_A-12':
		        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
		    'albert_en_base':
		        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
		    'electra_small':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'electra_base':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'experts_pubmed':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'experts_wiki_books':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		    'talking-heads_base':
		        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
		}

		tfhub_handle_encoder = map_name_to_handle[bert_model_name]
		tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
		return tfhub_handle_encoder, tfhub_handle_preprocess

	def setModel(self, tfhub_handle_preprocess, tfhub_handle_encoder) :
		bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess)
		bert_model = hub.KerasLayer(self.tfhub_handle_encoder)
		return bert_preprocess_model, bert_model
	
	def preprocessData(self, bert_preprocess_model) :
		pertanyaan_preprocessed = bert_preprocess_model([self.pertanyaan])
		jawaban_preprocessed = bert_preprocess_model(self.jawaban)
		return pertanyaan_preprocessed, jawaban_preprocessed

	def wordEmbedding(self, bert_model, pertanyaan_preprocessed, jawaban_preprocessed) :
		pertanyaan_results = bert_model(pertanyaan_preprocessed)
		jawaban_results = bert_model(jawaban_preprocessed)
		return pertanyaan_results['sequence_output'], jawaban_results['sequence_output']


class W2vEmbedding() :

	def __init__(self, pertanyaan, jawaban) :
		
		# Set Pertanyaan & Jawaban
		self.pertanyaan = pertanyaan
		self.jawaban = jawaban

		# Data Preprocessing
		pertanyaan_padded, jawaban_padded, tokenizer, vocab_size = self.dataPreprocessing(pertanyaan, jawaban)

		# Build Vocabulary
		vocab = self.buildVocabulary(pertanyaan_padded, jawaban_padded, tokenizer)

		# Set Embedding Size
		embed_size = 300

		# Train Model
		word_vectors = self.modelTraining(vocab, embed_size)

		# Create Embedding Matrix
		embedding_matrix = self.embeddingMatrix(vocab_size, embed_size, tokenizer, word_vectors)

		# Encode
		self.pertanyaan_results, self.jawaban_results = self.getEmbedded(embedding_matrix, pertanyaan_padded, jawaban_padded)

	def dataPreprocessing(self, pertanyaan, jawaban) :
		
		# Case Folding
		pertanyaan_cased = pertanyaan.lower()
		jawaban_cased = [i.lower() for i in jawaban]

		# Create Remover
		factory = StopWordRemoverFactory()
		remover = factory.create_stop_word_remover()

		# Stopword Removal
		pertanyaan_filtered = remover.remove(pertanyaan_cased)
		jawaban_filtered = [remover.remove(i) for i in jawaban_cased]

		# Filter Whitespace and Alphanumeric
		pertanyaan_filtered = re.sub(r'[^\w\s]','', pertanyaan_filtered)
		jawaban_filtered = [re.sub(r'[^\w\s]','', i) for i in jawaban_filtered]

		# Filter Number
		pertanyaan_filtered = re.sub(r'[\d]','', pertanyaan_filtered)
		jawaban_filtered = [re.sub(r'[\d]','', i) for i in jawaban_filtered]

		# Create Stemmer
		factory = StemmerFactory()
		stemmer = factory.create_stemmer()

		# Stemming
		pertanyaan_stemmed = stemmer.stem(pertanyaan_filtered)
		jawaban_stemmed = [stemmer.stem(i) for i in jawaban_filtered]

		# Create Corpus
		corpus = [pertanyaan_stemmed] + jawaban_stemmed

		# Create Tokenizer
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(corpus)

		# Tokenization & Vectorization
		pertanyaan_tokenized = tokenizer.texts_to_sequences([pertanyaan_stemmed])
		jawaban_tokenized = tokenizer.texts_to_sequences(jawaban_stemmed)
		
		# Get Vocabulary Length
		vocab_size = len(tokenizer.word_index) + 1

		# Get Max Length (For Padding)
		max_length = 0
		for i in jawaban_tokenized :
		  if len(i) > max_length :
		    max_length = len(i)

		# Padding
		pertanyaan_padded = pad_sequences(pertanyaan_tokenized, maxlen = max_length, padding = "post")
		jawaban_padded = pad_sequences(jawaban_tokenized, maxlen = max_length, padding = "post")
		
		return pertanyaan_padded, jawaban_padded, tokenizer, vocab_size

	def buildVocabulary(self, pertanyaan_padded, jawaban_padded, tokenizer) :

		# Vocabulary Initialization
		vocab = []

		# Build Vocabulary (Jawaban)
		for i in jawaban_padded :
		  vocab_now = [tokenizer.index_word[j] for j in i if (j != 0)]
		  vocab.append(vocab_now)

		# Build Vocabulary (Pertanyaan)
		vocab_now = [tokenizer.index_word[j] for j in pertanyaan_padded[0] if (j != 0)]
		vocab.append(vocab_now)

		return vocab

	def modelTraining(self, vocab, embed_size) :
		
		# Train Word2Vec
		word2vec_model = Word2Vec(vocab, vector_size = embed_size, window = 3, workers = 3, sg = 1, hs = 1, min_count = 0)

		# Set Word Vectors
		word_vectors = word2vec_model.wv

		return word_vectors

	def embeddingMatrix(self, vocab_size, embed_size, tokenizer, word_vectors) :
		
		# Initilization with numpy zeros
		embedding_matrix = np.zeros((vocab_size, embed_size))

		# Assign every word vectors in tokenizer to embedding matrix
		for word, i in tokenizer.word_index.items():
		    try :
		      embedding_vector = word_vectors.get_vector(word)
		      if embedding_vector is not None:
		          embedding_matrix[i] = embedding_vector
		    except :
		      print(word, "not in vocabulary")

		return embedding_matrix

	def getEmbedded(self, embedding_matrix, pertanyaan_padded, jawaban_padded) :
		# Embed Pertanyaan
		pertanyaan_results = np.array([embedding_matrix[i] for i in pertanyaan_padded])

		# Embed Jawaban
		jawaban_results = []
		for i in jawaban_padded :
		  embed_now = [embedding_matrix[j] for j in i]
		  jawaban_results.append(embed_now)
		jawaban_results = np.array(jawaban_results)

		return pertanyaan_results, jawaban_results

class PretrainedW2vEmbedding() :

	def __init__(self, pertanyaan, jawaban) :
		
		# Set Pertanyaan & Jawaban
		self.pertanyaan = pertanyaan
		self.jawaban = jawaban

		# Data Preprocessing
		pertanyaan_padded, jawaban_padded, tokenizer, vocab_size = self.dataPreprocessing(pertanyaan, jawaban)

		# Get Pre-train Word2Vec
		word2vec_model = "static/files/pretrainedw2vid/wiki.id.case.vector"
		word_vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=False)

		# Get Embedding Size
		embed_size = 400

		# Create Embedding Matrix
		embedding_matrix = self.embeddingMatrix(vocab_size, embed_size, tokenizer, word_vectors)

		# Encode
		self.pertanyaan_results, self.jawaban_results = self.getEmbedded(embedding_matrix, pertanyaan_padded, jawaban_padded)

	def dataPreprocessing(self, pertanyaan, jawaban) :
		
		# Case Folding
		pertanyaan_cased = pertanyaan.lower()
		jawaban_cased = [i.lower() for i in jawaban]

		# Create Remover
		factory = StopWordRemoverFactory()
		remover = factory.create_stop_word_remover()

		# Stopword Removal
		pertanyaan_filtered = remover.remove(pertanyaan_cased)
		jawaban_filtered = [remover.remove(i) for i in jawaban_cased]

		# Filter Whitespace and Alphanumeric
		pertanyaan_filtered = re.sub(r'[^\w\s]','', pertanyaan_filtered)
		jawaban_filtered = [re.sub(r'[^\w\s]','', i) for i in jawaban_filtered]

		# Filter Number
		pertanyaan_filtered = re.sub(r'[\d]','', pertanyaan_filtered)
		jawaban_filtered = [re.sub(r'[\d]','', i) for i in jawaban_filtered]

		# Create Stemmer
		factory = StemmerFactory()
		stemmer = factory.create_stemmer()

		# Stemming
		pertanyaan_stemmed = stemmer.stem(pertanyaan_filtered)
		jawaban_stemmed = [stemmer.stem(i) for i in jawaban_filtered]

		# Create Corpus
		corpus = [pertanyaan_stemmed] + jawaban_stemmed

		# Create Tokenizer
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(corpus)

		# Tokenization & Vectorization
		pertanyaan_tokenized = tokenizer.texts_to_sequences([pertanyaan_stemmed])
		jawaban_tokenized = tokenizer.texts_to_sequences(jawaban_stemmed)
		
		# Get Vocabulary Length
		vocab_size = len(tokenizer.word_index) + 1

		# Get Max Length (For Padding)
		max_length = 0
		for i in jawaban_tokenized :
		  if len(i) > max_length :
		    max_length = len(i)

		# Padding
		pertanyaan_padded = pad_sequences(pertanyaan_tokenized, maxlen = max_length, padding = "post")
		jawaban_padded = pad_sequences(jawaban_tokenized, maxlen = max_length, padding = "post")
		
		return pertanyaan_padded, jawaban_padded, tokenizer, vocab_size

	def buildVocabulary(self, pertanyaan_padded, jawaban_padded, tokenizer) :

		# Vocabulary Initialization
		vocab = []

		# Build Vocabulary (Jawaban)
		for i in jawaban_padded :
		  vocab_now = [tokenizer.index_word[j] for j in i if (j != 0)]
		  vocab.append(vocab_now)

		# Build Vocabulary (Pertanyaan)
		vocab_now = [tokenizer.index_word[j] for j in pertanyaan_padded[0] if (j != 0)]
		vocab.append(vocab_now)

		return vocab

	def modelTraining(self, vocab, embed_size) :
		
		# Train Word2Vec
		word2vec_model = Word2Vec(vocab, vector_size = embed_size, window = 3, workers = 3, sg = 1, hs = 1, min_count = 0)

		# Set Word Vectors
		word_vectors = word2vec_model.wv

		return word_vectors

	def embeddingMatrix(self, vocab_size, embed_size, tokenizer, word_vectors) :
		
		# Initilization with numpy zeros
		embedding_matrix = np.zeros((vocab_size, embed_size))

		# Assign every word vectors in tokenizer to embedding matrix
		for word, i in tokenizer.word_index.items():
		    try :
		      embedding_vector = word_vectors.get_vector(word)
		      if embedding_vector is not None:
		          embedding_matrix[i] = embedding_vector
		    except :
		      print(word, "not in vocabulary")

		return embedding_matrix

	def getEmbedded(self, embedding_matrix, pertanyaan_padded, jawaban_padded) :
		# Embed Pertanyaan
		pertanyaan_results = np.array([embedding_matrix[i] for i in pertanyaan_padded])

		# Embed Jawaban
		jawaban_results = []
		for i in jawaban_padded :
		  embed_now = [embedding_matrix[j] for j in i]
		  jawaban_results.append(embed_now)
		jawaban_results = np.array(jawaban_results)

		return pertanyaan_results, jawaban_results		

class MeasureDistance() :

	def __init__(self, pertanyaan_results, jawaban_results) :
		# Distance list initialization
		self.dist_list = []
		
		# Get vector pertanyaan
		self.pertanyaan_vectors = np.array(pertanyaan_results).squeeze(axis = 0)
		
		# Measure distance
		self.measure(self.pertanyaan_vectors, jawaban_results)

		# Sort Distance (Ascending)
		self.idx_sorted = np.argsort(self.dist_list)

	def measure(self, pertanyaan_vectors, jawaban_results) :
		for i in jawaban_results :
		  dist = self.cosineSimilarity(self.avgPooling(pertanyaan_vectors), self.avgPooling(i))
		  self.dist_list.append(dist)

	def avgPooling(self, x) :
		return K.mean(K.constant(x), axis=1, keepdims=False)

	def cosineSimilarity(self, x, y) :
	    normalize_a = tf.nn.l2_normalize(x,0)        
	    normalize_b = tf.nn.l2_normalize(y,0)
	    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
	    return float(cos_similarity)

class SelectAnswers() :
	
	def __init__(self, idx_sorted) :

		# Set Percentage
		self.percentage = 20

		# n labeled
		self.n_labeled = int(self.percentage/100 * len(idx_sorted))

		# Auto set to 3
		if (self.n_labeled < 3) :
		  self.n_labeled = 3
		
		# Set the amount of data for each section
		self.n_each = int(self.n_labeled / 3)

		# If there is excess data
		self.extra_top, self.extra_bottom = self.getExtra(self.n_labeled)
		
		# Combine
		self.idx_labeled = self.getIdxCombination(idx_sorted)

	def getExtra(self, n_labeled) :
		exc = n_labeled % 3
		extra_top = 0
		extra_bottom = 0
		if (exc) > 0 :
		  extra_top += 1
		  if exc == 2 :
		    extra_bottom += 1
		return extra_top, extra_bottom

	def getIdxTop(self, idx_sorted) :
		return [idx_sorted[i] for i in range(self.n_each + self.extra_top)]

	def getIdxMiddle(self, idx_sorted) :
		start_idx = int((len(idx_sorted)/2) - (self.n_each/2))
		return [idx_sorted[start_idx + i] for i in range(self.n_each)]

	def getIdxBottom(self, idx_sorted) :
		return [np.flip(idx_sorted)[i] for i in range(self.n_each + self.extra_bottom)]

	def getIdxCombination(self, idx_sorted) :
		return self.getIdxTop(idx_sorted) + self.getIdxMiddle(idx_sorted) + self.getIdxBottom(idx_sorted)

class DefineData() :

	def __init__(self, labels, idx_labeled, idx_sorted, pertanyaan_vectors, jawaban_results) :
		# Training Data
		self.pertanyaan_train = np.array([pertanyaan_vectors for i in range(len(idx_labeled))])
		self.x_train = np.array([jawaban_results[i] for i in idx_labeled])
		self.y_train = np.array(labels)

		# Get Index of Testing Data
		self.idx_test = [i for i in idx_sorted if i not in idx_labeled]

		# Testing Data
		self.pertanyaan_test = np.array([pertanyaan_vectors for i in range(len(self.idx_test))])
		self.x_test = np.array([jawaban_results[i] for i in self.idx_test])

class AESModel() :

	def __init__(self, 
		pertanyaan_train, x_train,
		y_train, 
		pertanyaan_test, x_test,
		id_mahasiswa, jawaban,
		idx_labeled, idx_test) :
		
		# Set Model
		self.model = self.getCNNLSTMModel(pertanyaan_train = pertanyaan_train, x_train = x_train)

		# Train Model
		epochs = 10
		self.model.fit([pertanyaan_train, x_train],
                    y_train,
                    epochs = epochs)

		# Get Predicted Score
		pred, pred_int = self.prediction(y_train, pertanyaan_test, x_test)

		# Get id_mahasiswa
		id_mahasiswa = self.getIDMahasiswa(id_mahasiswa, idx_labeled, idx_test)

		# Get jawaban
		jawaban = self.getJawaban(jawaban, idx_labeled, idx_test)
		
		# Create Dataframe
		self.df_final = self.predictionDF(id_mahasiswa, jawaban, pred_int, pred)

	def getCNNLSTMModel(self,
		pertanyaan_train,
		x_train,
	    filters = 64, 
	    kernel_size = 3, 
	    lstm_units = 128, 
	    lstm_dropout = 0.4, 
	    lstm_recurrent_dropout = 0.4, 
	    dropout = 0.4) :
	  
		left_input = Input(pertanyaan_train[0].shape)
		right_input = Input(x_train[0].shape)

		conv_layer = Conv1D(filters = filters, kernel_size = kernel_size, padding='same', strides = 1)
		lstm_layer = LSTM(units = lstm_units, dropout = lstm_dropout, recurrent_dropout = lstm_recurrent_dropout, return_sequences=False)
		dropout_layer = Dropout(dropout)

		question_encoding = dropout_layer(lstm_layer(conv_layer(left_input)))
		answer_encoding = dropout_layer(lstm_layer(conv_layer(right_input)))

		# output_model = Lambda(cosine_distance, output_shape = cos_dist_output_shape)([question_encoding, answer_encoding])
		output_model = concatenate([question_encoding, answer_encoding])
		output_model = Dense(1)(output_model)

		model = Model(inputs = [left_input, right_input], outputs = [output_model])

		model.compile(loss = 'mse',
		            optimizer = 'adam',
		            metrics = ['mae', 'mse'])
		model.summary()

		return model

	def prediction(self, y_train, pertanyaan_test, x_test) :
		pred = [float(i) for i in self.model.predict([pertanyaan_test, x_test])]
		pred_int = [round(i) for i in pred]

		pred_train = list(y_train)
		pred_int_train = [round(i) for i in y_train]

		pred_new = pred_train + pred
		pred_int_new = pred_int_train + pred_int

		return pred_new, pred_int_new

	def getIDMahasiswa(self, id_mahasiswa, idx_labeled, idx_test) :
		id_mahasiswa_train = [id_mahasiswa[i] for i in idx_labeled]
		id_mahasiswa_test = [id_mahasiswa[i] for i in idx_test]
		id_mahasiswa_new = id_mahasiswa_train + id_mahasiswa_test
		return id_mahasiswa_new

	def getJawaban(self, jawaban, idx_labeled, idx_test) :
		jawaban_train = [jawaban[i] for i in idx_labeled]
		jawaban_test = [jawaban[i] for i in idx_test]
		jawaban_new = jawaban_train + jawaban_test
		return jawaban_new

	def predictionDF(self, id_mahasiswa, jawaban, pred_int, pred) :
		df_final = pd.DataFrame({
		    'id_mahasiswa': id_mahasiswa,
		    'jawaban': jawaban,
		    'score': pred_int,
		    'score (float)': pred
		}).sort_values('id_mahasiswa').reset_index(drop=True)
		return df_final

	def getIDMahasiswaSorted(self) :
		return self.df_final['id_mahasiswa'].tolist()

	def getJawabanSorted(self) :
		return self.df_final['jawaban'].tolist()

	def getScoreIntSorted(self) :
		return self.df_final['score'].tolist()

	def getScoreSorted(self) :
		return self.df_final['score (float)'].tolist()

	# def saveToCSV(self) :
	# 	path = "./static/files/data/export/"
	# 	self.df_final.to_excel(f"{path}/scoring_result.csv", index = False)
	
	# def saveToXLSX(self) :
	# 	path = "./static/files/data/export/"
	# 	# Export with Columns Width Adjustment
	# 	filename = f"{path}/scoring_result.csv"
	# 	dfs = {'df_final': self.df_final}

	# 	writer = pd.ExcelWriter(filename, engine='xlsxwriter')
	# 	for sheetname, df in dfs.items():  # loop through `dict` of dataframes
	# 	    df.to_excel(writer, sheet_name=sheetname, index = False)  # send df to writer
	# 	    worksheet = writer.sheets[sheetname]  # pull worksheet object
	# 	    for idx, col in enumerate(df):  # loop through all columns
	# 	        series = df[col]
	# 	        max_len = max((
	# 	            series.astype(str).map(len).max(),  # len of largest item
	# 	            len(str(series.name))  # len of column name/header
	# 	            )) + 1  # adding a little extra space
	# 	        worksheet.set_column(idx, idx, max_len)  # set column width
	# 	writer.save()