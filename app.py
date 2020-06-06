import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('salary_prediction.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_feature = [int(x) for x in request.form.values()]
    features = [np.array(input_feature)]
    prediction = model.predict(features)
    output = np.round(prediction[0], 2)
    return render_template('index.html', final_output='Predicted Salary in $ {}'.format(output))


if __name__ == "__main__":

    app.run(debug=True)
