from transformers import BertTokenizer, AutoModel
import pytorch

class BERTEmbedding :

    def __init__(self, pertanyaan, jawaban, njawaban) :
        # set tokenizer & model
        self.tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        self.model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

        # set pertanyaan & jawaban
        self.pertanyaan = pertanyaan
        self.jawaban = jawaban

        # encode
        self.encode()

        # embed
        self.embed()
    
    def encode(self) :
        self.encoded_jawaban = self.tokenizer(self.jawaban, padding = True, return_tensors="pt")
        self.encoding_length = len(self.encoded_jawaban['input_ids'][0])
        self.encoded_pertanyaan = self.tokenizer(self.pertanyaan, max_length = getEncodingLength(), padding= 'max_length', return_tensors='pt')

    def embed(self) :
        # pertanyaan
        self.embedded_pertanyaan = self.model(pytorch.LongTensor(self.encoded_pertanyaan['input_ids']).view(1,-1))
        
        # jawaban
        embedded_jawaban = []
        for i in range(self.njawaban) :
            x = model(pytorch.LongTensor(self.encoded_jawaban['input_ids'][i]).view(1,-1))
            embedded_jawaban.append(x)
        
        # convert to numpy
        self.embedded_pertanyaan = self.embedded_pertanyaan['last_hidden_state'].detach().numpy()
        embedded_jawaban_np = []
        for i in range(self.njawaban) :
            embedded_jawaban_np.append(embedded_jawaban[i]['last_hidden_state'].detach().numpy())
        embedded_jawaban_np = np.array(embedded_jawaban_np)
        self.embedded_jawaban = embedded_jawaban_np
    
    def getEncodingLength(self) :
        return len(self.encoded_answers['input_ids'][0])