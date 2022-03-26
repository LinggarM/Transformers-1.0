from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route('/aes', methods=['GET', 'POST'])
def aes():
	return render_template("aes.html")

if __name__ == "__main__":
	app.static_folder = 'static'
	app.run()