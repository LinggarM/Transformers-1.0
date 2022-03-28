from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dropout, Dense
from keras.layers.merge import concatenate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CNNLSTMModel :

  def __init__(self, embedding, conv_filter, kernel_size, lstm_units, lstm_drop, lstm_rec_drop, drop, loss_func, opt, eval_metrics) :
    self.model = self.getModel(embedding, conv_filter, kernel_size, lstm_units, lstm_drop, lstm_rec_drop, drop, loss_func, opt, eval_metrics)

  def getModel(self, embedding, conv_filter, kernel_size, lstm_units, lstm_drop, lstm_rec_drop, drop, loss_func, opt, eval_metrics) :
    
    left_input = Input(embedding.embedded_pertanyaan.shape)
    right_input = Input(embedding.embedded_jawaban[0].shape)

    conv_layer = Conv1D(filters= conv_filter, kernel_size=kernel_size, padding='same', strides=1)
    lstm_layer = LSTM(lstm_units, dropout=lstm_drop, recurrent_dropout=lstm_rec_drop, return_sequences=False)
    dropout_layer = Dropout(drop)

    question_encoding = dropout_layer(lstm_layer(conv_layer(left_input)))
    answer_encoding = dropout_layer(lstm_layer(conv_layer(right_input)))

    output_model = concatenate([question_encoding, answer_encoding])
    output_model = Dense(1, activation='relu')(output_model)

    model = Model(inputs=[left_input, right_input], outputs=[output_model])

    model.compile(loss = loss_func,
                  optimizer = opt,
                  metrics = eval_metrics)
    model.summary()

    return model
  
  def visualizeModel(self) :
    return tf.keras.utils.plot_model(self.model, to_file='my_model.png')
  
  def trainModel(self, x_train_pertanyaan, x_train_jawaban, y_train, epochs) :
    history = self.model.fit([x_train_pertanyaan, x_train_jawaban],
                                 y_train,
                                 epochs = epochs)
    self.history = history
    return history
  
  def saveModel(self, path, name) :
    self.model.save_weights(f"{path}/{name}")
  
  def loadModel(self, path, name) :
    self.model.load_weights(f"{path}/{name}")
  
  def predictScore(self, x_test_pertanyaan, x_test_jawaban) :
    score_predict = self.model.predict([x_test_pertanyaan, x_test_jawaban])
    score_predict = [float(i[0]) for i in score_predict]
    score_predict = [round(i) for i in score_predict]
    return score_predict
  
  def getPredictedDF(self, id_mahasiswa, jawaban_list, idx_test, x_test_pertanyaan, x_test_jawaban) :
    df_final = pd.DataFrame({'id_mahasiswa': [id_mahasiswa[i] for i in idx_test], 'jawaban': [jawaban_list[i] for i in idx_test], 'nilai': self.predictScore(x_test_pertanyaan, x_test_jawaban)})
    return df_final
  
  def getLossVisualization(self) :
    plt.figure(figsize=(10, 4.5))
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(x = np.arange(len(self.history.history['loss'])), y = self.history.history['loss'], palette='hls')