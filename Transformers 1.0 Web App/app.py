from flask import Flask, request, render_template
from transformers import GlobalVariables, Data, BERTEmbedding, W2vEmbedding, PretrainedW2vEmbedding, MeasureDistance, SelectAnswers, DefineData, AESModel
import numpy as np

# Variables Initialization
globalVar = GlobalVariables()

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

		# Save to Global Variables
		globalVar.set_id_mahasiswa(id_mahasiswa)
		globalVar.set_jawaban(jawaban)

		# print(data.pertanyaan)
		# print(data.id_mahasiswa)
		# print(data.jawaban)
		# print(data.njawaban)

		# BERT Embedding
		embedding = BERTEmbedding(pertanyaan, jawaban)
		# print(embedding.pertanyaan_results)
		# print(embedding.jawaban_results[0])

		# # Word2vec Embedding
		# embedding = W2vEmbedding(pertanyaan, jawaban)
		
		# # Pre-trained Word2vec Embedding
		# embedding = PretrainedW2vEmbedding(pertanyaan, jawaban)

		# Measure Distance
		distance = MeasureDistance(embedding.pertanyaan_results, embedding.jawaban_results)
		# print(distance.idx_sorted)
		# print(distance.pertanyaan_vectors)

		# Select Answers
		selected = SelectAnswers(distance.idx_sorted)
		# print(selected.idx_labeled)

		# Jawaban to be Labeled
		jawaban_labeled = [jawaban[i] for i in selected.idx_labeled]

		# Get Iterator
		num_score = np.arange(len(jawaban_labeled))

		# Save to Global Variables
		globalVar.set_idx_sorted(distance.idx_sorted)
		globalVar.set_idx_labeled(selected.idx_labeled)
		globalVar.set_pertanyaan_vectors(distance.pertanyaan_vectors)
		globalVar.set_jawaban_results(embedding.jawaban_results)

		# Load Page
		return render_template('scoring.html', method = "post", jawaban_labeled = jawaban_labeled, num_score = num_score)
	
	# # Dummy Test
	# jawaban_labeled = ['cuk', 'cik', 'cak', 'cok']

	# # Get Iterator
	# num_score = np.arange(len(jawaban_labeled))

	# Load Page
	return render_template("scoring.html")
		# jawaban_labeled = jawaban_labeled,
		# num_score = num_score

@app.route('/result', methods=['GET', 'POST'])
def result():
	if request.method == 'POST':

		# Get Label
		labels = []
		num_score = np.arange(len(globalVar.idx_labeled))
		for i in num_score :
			name_now = f"jawaban_{i}"
			score_now = request.form[name_now]
			labels.append(int(score_now))

		# print(request.form['jawaban_0'])
		# print(request.form['jawaban_1'])
		# print(request.form['jawaban_2'])
		# print(labels)
		# print(globalVar.idx_labeled)
		# print(globalVar.idx_sorted)
		# print(globalVar.pertanyaan_vectors)
		# print(globalVar.jawaban_results)

		# Define Data
		definedata = DefineData(labels, globalVar.idx_labeled, globalVar.idx_sorted, globalVar.pertanyaan_vectors, globalVar.jawaban_results)
		# print(definedata.pertanyaan_train)
		# print(definedata.x_train)

		# Get Model & Train
		model = AESModel(
			definedata.pertanyaan_train, definedata.x_train,
			definedata.y_train, 
			definedata.pertanyaan_test, definedata.x_test,
			globalVar.id_mahasiswa, globalVar.jawaban,
			globalVar.idx_labeled, definedata.idx_test
		)

		# Get Prediction
		id_mahasiswa_new = model.getIDMahasiswaSorted()
		jawaban_new = model.getJawabanSorted()
		score_new = model.getScoreIntSorted()

		# Get Iterator
		num_score = np.arange(len(id_mahasiswa_new))

		# # Save Scoring Result
		# model.saveToCSV()
		# model.saveToXLSX()

		# Load Page
		return render_template('result.html', method = "post",
			id_mahasiswa_new = id_mahasiswa_new,
			jawaban_new = jawaban_new,
			score_new = score_new,
			num_score = num_score)
	
	# # Dummy Test
	# id_mahasiswa_new = [0, 3, 2, 1]
	# jawaban_new = ['cuk', 'cik', 'cak', 'cok']
	# score_new = [10, 9, 9, 10]

	# # Get Iterator
	# num_score = np.arange(len(id_mahasiswa_new))

	# Load Page
	return render_template("result.html")
			# id_mahasiswa_new = id_mahasiswa_new,
			# jawaban_new = jawaban_new,
			# score_new = score_new,
			# num_score = num_score

@app.route('/tutorial', methods=['GET', 'POST'])
def tutorial():
	return render_template("tutorial.html")

@app.route('/about', methods=['GET', 'POST'])
def about():
	return render_template("about.html")

@app.route('/test', methods=['GET', 'POST'])
def test():
	return render_template("test.html")

if __name__ == "__main__":
	app.static_folder = 'static'
	app.run()