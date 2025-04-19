from flask import Flask, render_template, request
import pickle

# Load vectorizer and model
with open('Sentiment_Analysis_Prediction1.pkl', 'rb') as file:
    vector, model = pickle.load(file)

# Mapping numerical output to labels
label_map = {
    0: "Negative",
    1: "Positive" # if applicable
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vector.transform([user_input])
        raw_prediction = model.predict(transformed_input)[0]
        prediction = label_map.get(raw_prediction, "Unknown")  # Convert number to label
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
