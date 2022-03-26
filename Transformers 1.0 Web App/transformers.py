import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk import word_tokenize
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout, Activation
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import L2
from keras.models import load_model
nltk.download('punkt')

from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import csv

def case_folding(sentence) :
  sentence = sentence.lower() #mengubah menjadi lower case
  sentence_nontb = re.sub(r'[^\w\s]','', sentence) #membuang karakter selain whitespace dan alphanumeric
  sentence_nonnumber = re.sub(r'[\d]','', sentence_nontb) #membuang karakter angka
  return sentence_nonnumber

def stopword_remover(sentence) :
  factory = StopWordRemoverFactory()
  stopword = factory.create_stop_word_remover()
  stop = stopword.remove(sentence)
  return stop

def stemmer_indonesia(sentence) :
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  stem_result = stemmer.stem(sentence)
  return stem_result

def tokenizer_indonesia(sentence) :
  tokens = nltk.tokenize.word_tokenize(sentence)
  return tokens

def count_padding(dataframe) :
    pad_length = 0
    for i in range(len(dataframe)) :
      sen_length = len(dataframe['review'][i])
      if (sen_length > pad_length) :
        pad_length = sen_length
    return pad_length

def apply_padding(sentence, pad_length) :
  panjang = len(sentence)
  if (panjang > pad_length) :
    padded = sentence[:pad_length]
  else :
    padded = sentence
    for i in range(pad_length - panjang) :
      padded.append('<pad>')
  return padded

def vektorizer(dataframe) :
  max_words = 100000
  tokenizer = Tokenizer(num_words= max_words, lower= True)
  tokenizer.fit_on_texts(dataframe['review'])
  sequences = tokenizer.texts_to_sequences(dataframe['review'])
  return sequences, tokenizer

def preprocess(dataframe) :
  for index, row in dataframe.iterrows():
    dataframe.at[index, 'review'] = case_folding(row[0])

  for index, row in dataframe.iterrows():
    dataframe.at[index, 'review'] = stopword_remover(row[0])

  for index, row in dataframe.iterrows():
    dataframe.at[index, 'review'] = stemmer_indonesia(row[0])

  for index, row in dataframe.iterrows():
    dataframe.at[index, 'review'] = tokenizer_indonesia(row[0])

  pad_length = count_padding(dataframe) #menghitung panjang review dengan teks terpanjang
  for index, row in dataframe.iterrows():
    dataframe.at[index, 'review'] = apply_padding(row[0], pad_length)
  
  sequences, tokenizer = vektorizer(dataframe)

  return sequences, tokenizer


# Load Vectorizer
with open('static/files/vectorizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Load Word2vec Model
with open('static/files/model_word2vec.pickle', 'rb') as handle:
    model_word2vec = pickle.load(handle)


# Load Embedding_Matrix
with open('static/files/embedding_matrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)


max_words = 100000
embed_size = 300
max_length = 398
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)


from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)


def get_model() :
  model = Sequential()
  model.add(Input(shape = (max_length,)))
  model.add(Embedding(num_words, embed_size, weights= [embedding_matrix], trainable=False))
  model.add(LSTM(128, return_sequences= True, dropout= 0.3, recurrent_dropout= 0.3))
  model.add(Attention(return_sequences= False))
  model.add(Dense(1200, activation='tanh'))
  model.add(Dropout(0.2))
  model.add(Dense(600, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(15, activation='sigmoid'))
  model.load_weights('static/files/model_weights.h5')
  return model


def analyze_from_sentence(sentence) :
  model = get_model()
  s = case_folding(sentence)
  s = stopword_remover(s)
  s = stemmer_indonesia(s)
  s = tokenizer_indonesia(s)
  s = apply_padding(s, pad_length = 398)
  seq = tokenizer.texts_to_sequences([s])

  if (len(seq[0]) < 398) :
    tambah = 398 - len(seq[0])
    for i in range(tambah) :
      seq[0].append(1)

  predicted = model.predict(np.array(seq))
  predicted[predicted>=0.5] = 1
  predicted [predicted<0.5] = 0
  return predicted
 

def predicted_to_text(predicted_vector):
  predicted = predicted_vector.astype(int)
  predicted = predicted[0]

  result_text = ""

  # Makanan
  if (predicted[0:3] == np.array([1, 0, 0])).all() or (predicted[0:3] == np.array([0, 0, 1])).all() or (predicted[0:3] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[0:3] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Makanan Positif, "
    elif (predicted[0:3] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Makanan Negatif, "
  else :
    result_text = result_text + "Makanan Netral, "

  # Kamar
  if (predicted[3:6] == np.array([1, 0, 0])).all() or (predicted[3:6] == np.array([0, 0, 1])).all() or (predicted[3:6] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[3:6] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Kamar Positif, "
    elif (predicted[3:6] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Kamar Negatif, "
  else :
    result_text = result_text + "Kamar Netral, "

  # Pelayanan
  if (predicted[6:9] == np.array([1, 0, 0])).all() or (predicted[6:9] == np.array([0, 0, 1])).all() or (predicted[6:9] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[6:9] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Pelayanan Positif, "
    elif (predicted[6:9] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Pelayanan Negatif, "
  else :
    result_text = result_text + "Pelayanan Netral, "

  # Lokasi
  if (predicted[9:12] == np.array([1, 0, 0])).all() or (predicted[9:12] == np.array([0, 0, 1])).all() or (predicted[9:12] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[9:12] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Lokasi Positif, "
    elif (predicted[9:12] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Lokasi Negatif, "
  else :
    result_text = result_text + "Lokasi Netral, "

  # Lain
  if (predicted[12:15] == np.array([1, 0, 0])).all() or (predicted[12:15] == np.array([0, 0, 1])).all() or (predicted[12:15] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[12:15] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Lain Positif, "
    elif (predicted[12:15] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Lain Negatif, "
  else :
    result_text = result_text + "Lain Netral, "

  # Buang Koma di Belakang
  if (result_text[-2:] == ", ") :
    result_text = result_text[:-2]
  
  return result_text


def predicted_to_text_and_value(predicted_vector):
  predicted = predicted_vector.astype(int)
  predicted = predicted[0]

  result_text = ""

  makanan_positif = makanan_negatif = makanan_netral = kamar_positif = kamar_negatif = kamar_netral = pelayanan_positif = pelayanan_negatif = pelayanan_netral = lokasi_positif = lokasi_negatif = lokasi_netral = lain_positif = lain_negatif = lain_netral = 0

  # Makanan
  if (predicted[0:3] == np.array([1, 0, 0])).all() or (predicted[0:3] == np.array([0, 0, 1])).all() or (predicted[0:3] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[0:3] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Makanan Positif, "
      makanan_positif+=1
    elif (predicted[0:3] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Makanan Negatif, "
      makanan_negatif+=1
  else :
    result_text = result_text + "Makanan Netral, "
    makanan_netral+=1

  # Kamar
  if (predicted[3:6] == np.array([1, 0, 0])).all() or (predicted[3:6] == np.array([0, 0, 1])).all() or (predicted[3:6] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[3:6] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Kamar Positif, "
      kamar_positif+=1
    elif (predicted[3:6] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Kamar Negatif, "
      kamar_negatif+=1
  else :
    result_text = result_text + "Kamar Netral, "
    kamar_netral+=1

  # Pelayanan
  if (predicted[6:9] == np.array([1, 0, 0])).all() or (predicted[6:9] == np.array([0, 0, 1])).all() or (predicted[6:9] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[6:9] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Pelayanan Positif, "
      pelayanan_positif+=1
    elif (predicted[6:9] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Pelayanan Negatif, "
      pelayanan_negatif+=1
  else :
    result_text = result_text + "Pelayanan Netral, "
    pelayanan_netral+=1

  # Lokasi
  if (predicted[9:12] == np.array([1, 0, 0])).all() or (predicted[9:12] == np.array([0, 0, 1])).all() or (predicted[9:12] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[9:12] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Lokasi Positif, "
      lokasi_positif+=1
    elif (predicted[9:12] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Lokasi Negatif, "
      lokasi_negatif+=1
  else :
    result_text = result_text + "Lokasi Netral, "
    lokasi_netral+=1

  # Lain
  if (predicted[12:15] == np.array([1, 0, 0])).all() or (predicted[12:15] == np.array([0, 0, 1])).all() or (predicted[12:15] == np.array([0, 0, 0])).all() :
    # Jika Positif, Negatif, atau None
    if (predicted[12:15] == np.array([1, 0, 0])).all() :
      result_text = result_text + "Lain Positif, "
      lain_positif+=1
    elif (predicted[12:15] == np.array([0, 0, 1])).all() :
      result_text = result_text + "Lain Negatif, "
      lain_negatif+=1
  else :
    result_text = result_text + "Lain Netral, "
    lain_netral+=1

  # Buang Koma di Belakang
  if (result_text[-2:] == ", ") :
    result_text = result_text[:-2]
  
  return result_text, makanan_positif, makanan_negatif, makanan_netral, kamar_positif, kamar_negatif, kamar_netral, pelayanan_positif, pelayanan_negatif, pelayanan_netral, lokasi_positif, lokasi_negatif, lokasi_netral, lain_positif, lain_negatif, lain_netral


def analyze_from_file(file) :
  #get_file
  df_file = pd.read_csv(file, header=None, names=["review"])
  
  #initialization
  predicted_results = []
  total_makanan_positif = total_makanan_negatif = total_makanan_netral = total_kamar_positif = total_kamar_negatif = total_kamar_netral = total_pelayanan_positif = total_pelayanan_negatif = total_pelayanan_netral = total_lokasi_positif = total_lokasi_negatif = total_lokasi_netral = total_lain_positif = total_lain_negatif = total_lain_netral = 0
  df_food = pd.DataFrame({
      "review":[],
      "prediction":[]
  })
  df_room = pd.DataFrame({
      "review":[],
      "prediction":[]
  })
  df_service = pd.DataFrame({
      "review":[],
      "prediction":[]
  })
  df_location = pd.DataFrame({
      "review":[],
      "prediction":[]
  })
  df_other = pd.DataFrame({
      "review":[],
      "prediction":[]
  })
  
  for index, row in df_file.iterrows():
    #get results
    result_text, makanan_positif, makanan_negatif, makanan_netral, kamar_positif, kamar_negatif, kamar_netral, pelayanan_positif, pelayanan_negatif, pelayanan_netral, lokasi_positif, lokasi_negatif, lokasi_netral, lain_positif, lain_negatif, lain_netral = predicted_to_text_and_value(analyze_from_sentence(row[0]))
    
    #assign predicted text
    predicted_results.append(result_text)
    
    #assign sentiment
    total_makanan_positif = total_makanan_positif + makanan_positif
    total_makanan_negatif = total_makanan_negatif + makanan_negatif
    total_makanan_netral = total_makanan_netral + makanan_netral
    total_kamar_positif = total_kamar_positif + kamar_positif
    total_kamar_negatif = total_kamar_negatif + kamar_negatif
    total_kamar_netral = total_kamar_netral + kamar_netral
    total_pelayanan_positif = total_pelayanan_positif + pelayanan_positif
    total_pelayanan_negatif = total_pelayanan_negatif + pelayanan_negatif
    total_pelayanan_netral = total_pelayanan_netral + pelayanan_netral
    total_lokasi_positif = total_lokasi_positif + lokasi_positif
    total_lokasi_negatif = total_lokasi_negatif + lokasi_negatif
    total_lokasi_netral = total_lokasi_netral + lokasi_netral
    total_lain_positif = total_lain_positif + lain_positif
    total_lain_negatif = total_lain_negatif + lain_negatif
    total_lain_netral = total_lain_netral + lain_netral
	
	#dataframe per aspect
    if ((makanan_positif == 1) or (makanan_negatif == 1) or (makanan_netral == 1)) :
      df_food.loc[len(df_food.index)] = [row[0], result_text]
    if ((kamar_positif == 1) or (kamar_negatif == 1) or (kamar_netral == 1)) :
      df_room.loc[len(df_room.index)] = [row[0], result_text]
    if ((pelayanan_positif == 1) or (pelayanan_negatif == 1) or (pelayanan_netral == 1)) :
      df_service.loc[len(df_service.index)] = [row[0], result_text]
    if ((lokasi_positif == 1) or (lokasi_negatif == 1) or (lokasi_netral == 1)) :
      df_location.loc[len(df_location.index)] = [row[0], result_text]
    if ((lain_positif == 1) or (lain_negatif == 1) or (lain_netral == 1)) :
      df_other.loc[len(df_other.index)] = [row[0], result_text]
  
  #finalisasi output
  df_file['prediction'] = predicted_results
  review_no = np.arange(1, len(df_file)+1)
  df_file.insert(0, 'No', review_no)
  
  sentiment_list = [total_makanan_positif, total_makanan_negatif, total_makanan_netral, total_kamar_positif, total_kamar_negatif, total_kamar_netral, total_pelayanan_positif, total_pelayanan_negatif, total_pelayanan_netral, total_lokasi_positif, total_lokasi_negatif, total_lokasi_netral, total_lain_positif, total_lain_negatif, total_lain_netral]

  review_no = np.arange(1, len(df_food)+1)
  df_food.insert(0, 'No', review_no)
  review_no = np.arange(1, len(df_room)+1)
  df_room.insert(0, 'No', review_no)
  review_no = np.arange(1, len(df_service)+1)
  df_service.insert(0, 'No', review_no)
  review_no = np.arange(1, len(df_location)+1)
  df_location.insert(0, 'No', review_no)
  review_no = np.arange(1, len(df_other)+1)
  df_other.insert(0, 'No', review_no)

  return df_file, sentiment_list, df_food, df_room, df_service, df_location, df_other

def analyze_from_web(link) :
  link = link.replace("www", "m")
  chrome_options = ChromeOptions()
  chrome_options.add_argument("--start-maximized")
  chrome_options.add_argument("--headless")
  chrome_options.add_argument('--disable-gpu')
  chrome_options.add_argument('--disable-extensions')
  chrome_options.add_argument('--ignore-certificate-errors')
  chrome_options.add_argument('--ignore-ssl-errors')
  chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
  driver = webdriver.Chrome(options= chrome_options)
  driver.get(link)

  # semua ulasan
  nxt = driver.find_element(By.XPATH,'//div[@class="_1uXCv"]')
  element = driver.find_element(By.XPATH,'//div[@class="_3uVKw"]')
  actions = ActionChains(driver)
  actions.move_to_element(element).perform()
  nxt.click()

  # bahasa
  bahasa = driver.find_elements(By.XPATH,'//div[@class="_2TzAV _2qq6s _2WrKe"]')[1]
  tenger = driver.find_element(By.XPATH,'//div[@class="Ck-bZ"]')
  tenger.location_once_scrolled_into_view
  bahasa.click()
  time.sleep(2)
  bahasaIndonesia = driver.find_element(By.XPATH,'//span[text()="Bahasa Indonesia"]').click()

  # init data
  data = []

  # open review (n halaman x 10 review)
  n = 20 # jumlah halaman
  nxt2 = driver.find_element(By.XPATH,'//div[@class="_1hiwh"]')
  for i in range(n):
    time.sleep(2)
    nxt2.location_once_scrolled_into_view
    nxt2.click()

  # parse html
  html = driver.page_source
  soup = BeautifulSoup(html, "html.parser")
  divs = soup.find_all("div", "css-901oao r-1sixt3s r-1b43r93 r-majxgm r-rjixqe r-fdjqy7")

  # get data
  for div in divs:
    try :
      if ((len(div.string) > 20) and (not div.string.startswith("Kepada "))) :
        data.append(div.string)
    except :
      print("Bukan teks!")

  # save file
  with open("static/files/reviews_web.csv", "w+", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for item in data :
      writer.writerow([item])

  driver.close()

def preprocessingdata(sentence) :
  model = get_model()
  s = case_folding(sentence)
  s = stopword_remover(s)
  s = stemmer_indonesia(s)
  s = tokenizer_indonesia(s)
  s = apply_padding(s, pad_length = 398)
  seq = tokenizer.texts_to_sequences([s])
  return seq