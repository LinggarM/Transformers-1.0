from flask import Flask, request, render_template
from transformers import Data, BERTEmbedding, MeasureDistance, SelectAnswers, DefineData, AESModel
import numpy as np

# Variables Initialization
id_mahasiswa = []
jawaban = []
idx_labeled = []
idx_sorted = []
pertanyaan_vectors = np.empty([2, 2])
jawaban_results = np.empty([2, 2])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route('/aes', methods=['GET', 'POST'])
def aes():
	return render_template("aes.html")

@app.route('/scoring', methods=['GET', 'POST'])
def scoring():
	if request.method == 'POST':
		
		# Save Pertanyaan
		pertanyaan = request.form['pertanyaan']
		open('static/files/data/pertanyaan.txt', "w").write(pertanyaan)

		# Save Jawaban
		f = request.files['file']
		f.save('static/files/data/jawaban.csv')

		# Load Data
		data = Data()
		id_mahasiswa = data.id_mahasiswa
		pertanyaan = data.pertanyaan
		jawaban = data.jawaban

		# print(data.pertanyaan)
		# print(data.id_mahasiswa)
		# print(data.jawaban)
		# print(data.njawaban)

		# Word Embedding
		embedding = BERTEmbedding(pertanyaan, jawaban)
		print(embedding.pertanyaan_results)
		print(embedding.jawaban_results[0])

		# Measure Distance
		distance = MeasureDistance(embedding.pertanyaan_results, embedding.jawaban_results)
		print(distance.idx_sorted)
		print(distance.pertanyaan_vectors)

		# Select Answers
		selected = SelectAnswers(distance.idx_sorted)
		print(selected.idx_labeled)

		# Jawaban to be Labeled
		jawaban_labeled = [jawaban[i] for i in selected.idx_labeled]
		num_score = np.arange(len(jawaban_labeled))

		# Save variable to be used in the next page
		idx_labeled = distance.idx_sorted
		idx_sorted = selected.idx_labeled
		pertanyaan_vectors = distance.pertanyaan_vectors
		jawaban_results = embedding.jawaban_results

		# Load Page
		return render_template('scoring.html', method = "post", jawaban_labeled = jawaban_labeled, num_score = num_score)
	
	# Load Page
	return render_template("scoring.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
	if request.method == 'POST':

		# Get Label
		labels = []
		num_score = np.arange(len(idx_labeled))
		for i in num_score :
			name_now = f"jawaban_{i}"
			score_now = request.form[name_now]
			labels.append(int(score_now))

		# Define Data
		definedata = DefineData(labels, idx_labeled, idx_sorted, pertanyaan_vectors, jawaban_results)

		# Get Model & Train
		model = AESModel(
			definedata.pertanyaan_train, definedata.x_train,
			definedata.y_train, 
			definedata.pertanyaan_test, definedata.x_test,
			id_mahasiswa, jawaban,
			idx_labeled, definedata.idx_test
		)

		# Get Prediction
		id_mahasiswa_new = model.getIDMahasiswaSorted()
		jawaban_new = model.getJawabanSorted()
		score_new = model.getScoreIntSorted()

		# Save Scoring Result
		model.saveToCSV()
		model.saveToXLSX()

		# Load Page
		return render_template('result.html', method = "post",
			id_mahasiswa_new = id_mahasiswa_new,
			jawaban_new = jawaban_new,
			score_new = score_new)
	
	# Get Prediction
	id_mahasiswa_new = [0, 3, 2, 1]
	jawaban_new = ['cuk', 'cik', 'cak', 'cok']
	score_new = [10, 9, 9, 10]
	num_score = np.arange(len(id_mahasiswa_new))

	# Load Page
	return render_template("result.html",
			id_mahasiswa_new = id_mahasiswa_new,
			jawaban_new = jawaban_new,
			score_new = score_new,
			num_score = num_score)

@app.route('/tutorial', methods=['GET', 'POST'])
def tutorial():
	return render_template("tutorial.html")

@app.route('/about', methods=['GET', 'POST'])
def about():
	return render_template("about.html")

if __name__ == "__main__":
	app.static_folder = 'static'
	app.run()