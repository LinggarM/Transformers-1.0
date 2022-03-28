import numpy as np
import pandas as pd

class Data :

    def __init__(self) :
        # path
        self.path = "data"
        self.path_pertanyaan = f"{self.path}/pertanyaan.txt"
        self.path_jawaban = f"{self.path}/jawaban.csv"

        # pertanyaan
        self.pertanyaan = open(self.path_pertanyaan).read()

        # jawaban
        self.jawaban_df = pd.read_csv(self.path_jawaban)
        self.jawaban_list = self.jawaban_df['jawaban'].tolist()

        # id_mahasiswa
        self.id_mahasiswa = self.jawaban_df['id_mahasiswa'].tolist()

        # n jawaban
        self.njawaban = len(self.jawaban_df)