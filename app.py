from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer



model=pickle.load(open('./model/nbc_model.pickle','rb'))
cv=pickle.load(open('./model/counter_vector.pkl','rb'))
def new_prediction(new_review):
    review=re.sub('[^a-zA-Z]',' ',new_review)
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    #all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if word not in set(all_stopwords)]
    review=" ".join(review)
    review=[review]
    new_X=cv.transform(review).toarray()
    new_X=new_X.reshape(1,new_X.size)
    p=model.predict(new_X)
    return p[0]


app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        review=request.form['reviews']
        p=new_prediction(review)
        if p==1:
            return render_template('index.html',msg='This is a positive review',c='green')
        else:
            return render_template('index.html',msg='This is a negative review',c='red')

    else:
        return render_template('index.html')




if __name__=='__main__':
    app.run(debug=True)