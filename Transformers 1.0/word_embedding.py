from transformers import BertTokenizer, AutoModel
import torch
import numpy as np

class BERTEmbedding :

    def __init__(self, pertanyaan, jawaban, njawaban) :
        # set tokenizer & model
        self.tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        self.model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

        # encode
        self.encode(pertanyaan, jawaban)

        # embed
        self.embed(njawaban)
    
    def encode(self, pertanyaan, jawaban) :
        self.encoded_jawaban = self.tokenizer(jawaban, padding = True, return_tensors="pt")
        self.encoding_length = len(self.encoded_jawaban['input_ids'][0])
        self.encoded_pertanyaan = self.tokenizer(pertanyaan, max_length = self.encoding_length, padding= 'max_length', return_tensors='pt')

    def embed(self, njawaban) :
        # pertanyaan
        self.embedded_pertanyaan = self.model(torch.LongTensor(self.encoded_pertanyaan['input_ids']).view(1,-1))
        
        # jawaban
        embedded_jawaban = []
        for i in range(njawaban) :
            x = self.model(torch.LongTensor(self.encoded_jawaban['input_ids'][i]).view(1,-1))
            embedded_jawaban.append(x)
        
        # convert to numpy
        self.embedded_pertanyaan = self.embedded_pertanyaan['last_hidden_state'].detach().numpy()
        embedded_jawaban_np = []
        for i in range(njawaban) :
            embedded_jawaban_np.append(embedded_jawaban[i]['last_hidden_state'].detach().numpy())
        embedded_jawaban_np = np.array(embedded_jawaban_np)
        self.embedded_jawaban = embedded_jawaban_np

        # simplify dimension
        self.embedded_pertanyaan  = np.squeeze(self.embedded_pertanyaan, axis = 0)
        self.embedded_jawaban = np.squeeze(self.embedded_jawaban, axis = 1)