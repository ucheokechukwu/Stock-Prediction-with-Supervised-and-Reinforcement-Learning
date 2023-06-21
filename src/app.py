# import Flask and jsonify
from flask import Flask, render_template
import flask
from get_output import get_output

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/crystalball', methods=['GET'])
def predict():
    if flask.request.method == 'GET':
        output = get_output()
        return render_template('crystalball.html', output=output)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
