from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route('/aes', methods=['GET', 'POST'])
def aes():
	if request.method == 'POST':
		
		# Save Pertanyaan
		pertanyaan = request.form['pertanyaan']
		open('static/files/data/pertanyaan.txt', "w").write(pertanyaan)

		# Save Jawaban
		f = request.files['file']
		f.save('static/files/data/jawaban.csv')

	return render_template("aes.html")

@app.route('/tutorial', methods=['GET', 'POST'])
def tutorial():
	return render_template("tutorial.html")

@app.route('/about', methods=['GET', 'POST'])
def about():
	return render_template("about.html")

if __name__ == "__main__":
	app.static_folder = 'static'
	app.run()