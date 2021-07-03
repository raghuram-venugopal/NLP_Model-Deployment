from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.naive_bayes import MultinomialNB
import re
import unidecode


# load the model from disk
clf = pickle.load(open('NLP_Model.pkl', 'rb'))
tfidf=pickle.load(open('transform.pkl','rb'))
stopwords=pickle.load(open('stopwords.pkl','rb'))
f20words=pickle.load(open('freqwords.pkl','rb'))
rare20words=pickle.load(open('rarewords.pkl','rb'))
app = Flask(__name__)

def preprocess(text):
    sentence = str(text).lower()
    sentence = re.sub("[^A-Z a-z 0-9-]+",'',sentence)
    sentence = " ".join(sentence.split())
    sentence = unidecode.unidecode(sentence)
    sentence = " ".join([t for t in str(sentence).split() if t not in stopwords])
    #sentence = convert_to_base(sentence)
    sentence = " ".join([t for t in sentence.split() if t not in f20words])
    sentence = " ".join([t for t in sentence.split() if t not in rare20words])
    return sentence

@app.route('/')
def home():
	return render_template('home.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform([preprocess(data)]).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)