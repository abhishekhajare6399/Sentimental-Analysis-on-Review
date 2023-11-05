## streamlit run Sentimental_Analysis_App_Streamlit.py
import re
import streamlit as st
import pickle
import warnings
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

cv = pickle.load(open("model/count-Vectorizer-Model.pkl" , "rb"))
model = pickle.load(open("model/Movies_Review_Classification_Model.pkl" , "rb")) 
print('model Loaded')

cv1 = pickle.load(open("model/count-Vectorizer-Product.pkl" , "rb"))
model1 = pickle.load(open("model/Product_Review_Classification1.pkl" , "rb")) 
print('model Loaded')

# SVMcv = pickle.load(open("model/count-Vectorizer-Product.pkl" , "rb"))
# SVM = pickle.load(open("model/Product_Review_Classification1.pkl" , "rb")) 

Randomcv = pickle.load(open("model/count-Vectorizer-Random.pkl" , "rb"))
Random = pickle.load(open("model/Movies_Review_Random.pkl" , "rb")) 

logistic = pickle.load(open("model/Movies_Review_Logistic.pkl" , "rb"))
def test_model(sentence):
    sen = cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        resu = 'Positive Reviews'
        return resu
    else:
        resu = 'Negative Reviews'
        return resu
    
def test_model1(sentence):
    sen = cv.transform([sentence]).toarray()
    res = logistic.predict(sen)[0]
    if res == 1:
        resu = 'Positive Reviews'
        return resu
    else:
        resu = 'Negative Reviews'
        return resu
    
def test_model2(sentence):
    sen = Randomcv.transform([sentence]).toarray()
    res = Random.predict(sen)[0]
    if res == 'positive':
        resu = 'Positive Reviews'
        return resu
    else:
        resu = 'Negative Reviews'
        return resu
    
def test_model4(sentence):
    sen = cv1.transform([sentence]).toarray()
    res = model1.predict(sen)[0]
    if res == 0 or res == 1:
        resu = 'Negative Reviews'
        return resu
    elif res == 2:
        resu = 'Average Reviews'
        return resu
    else:
        resu = 'Postive Reviews'
        return resu



def main():
    st.title("Machine Learning Project")
    activities = ["Sentimental Analysis"]
    choice = st.sidebar.selectbox("Select Activity", ["Sentimental Analysis For Movies","Sentimental Analysis For Product"])

    if choice == 'Sentimental Analysis For Movies':
        st.subheader("Sentimental Analysis For Movie Review")
        article_text = st.text_area("Enter Review Here","Type your reaction here")
       

        summary_choice = st.selectbox("Algorithm Choice" , ["Naive Bayes","LogisticRegression","Random Forest"])
        if st.button("Predict Review"):
            l=len(article_text.split(' '))
            ps = PorterStemmer()
            article_text = re.sub("[^a-zA-Z]"," ", article_text)
            article_text = article_text.lower()
            article_text = article_text.split()
            article_text = [ps.stem(word) for word in article_text if word not in set(stopwords.words("english"))]
            article_text = " ".join(article_text)
            print(article_text)
            if summary_choice == 'Naive Bayes':
                summary_result = test_model(article_text)
            elif summary_choice == 'LogisticRegression':
                summary_result = test_model1(article_text)
            elif summary_choice == 'Random Forest':
                summary_result = test_model2(article_text)
            st.write(summary_result)

    if choice == 'Sentimental Analysis For Product':
        st.subheader("Sentimental Analysis For Product Review")
        article_text = st.text_area("Enter Review Here","Type your reaction here")
       

        summary_choice = st.selectbox("Algorithm Choice" , ["Naive Bayes"])
        if st.button("Predict Review"):
            l=len(article_text.split(' '))
            ps = PorterStemmer()
            article_text = re.sub("[^a-zA-Z]"," ", article_text)
            article_text = article_text.lower()
            article_text = article_text.split()
            article_text = [ps.stem(word) for word in article_text if word not in set(stopwords.words("english"))]
            article_text = " ".join(article_text)
            print(article_text)
            if summary_choice == 'Naive Bayes':
                summary_result = test_model(article_text)
            st.write(summary_result)
          


if __name__=='__main__':
	main()