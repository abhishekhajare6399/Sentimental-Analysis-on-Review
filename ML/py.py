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

cv = pickle.load(open("model/count-Vectorizer.pkl" , "rb"))
model = pickle.load(open("model/Movies_Review_Classification.pkl" , "rb")) 

cv1 = pickle.load(open("model/count-Vectorizer-Product.pkl" , "rb"))
model1 = pickle.load(open("model/Product_Review_Classification1.pkl" , "rb")) 

# SVMcv = pickle.load(open("model/count-Vectorizer-Product.pkl" , "rb"))
# SVM = pickle.load(open("model/Product_Review_Classification1.pkl" , "rb")) 

Randomcv = pickle.load(open("model/count-Vectorizer-Random.pkl" , "rb"))
Random = pickle.load(open("model/Movies_Review_Random.pkl" , "rb")) 
print('model Loaded')



