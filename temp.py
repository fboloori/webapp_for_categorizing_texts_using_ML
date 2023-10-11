import streamlit as st
import nltk, hazm
from nltk.stem import WordNetLemmatizer
from hazm import word_tokenize
lemmatizer = WordNetLemmatizer()

import pickle 
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd  
import re 

from sklearn.model_selection import train_test_split
from sklearn import svm
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

########################################################################################################################

class TrainTestModel:
    # This class builds a model for text mining either by cached files 
    # or by running a model based on the "model_name" given to "__init__" function.
    # Function "read_from_cached_files()" reads trained_model object and its depedent objects
    # (i.e. "self.trained_model , self.le, ...." from cached files if the files exist.
    # otherwise, it returns False and we must train our model using "self.train_model" function
    # Finally we save trained_model object and its vectorizer , ..... in files using "save_results_to_files()".

    def __init__(self,model_name):  
        self.model_name = model_name
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.le=None
        self.vectorizer=None
        self.trained_model=None
        self.confusion=None
        self.stopwords= self.stopwords_eng_fa()
        
        if self.read_from_cached_files()==False:
            self.train_model()
            self.save_results_to_files()

    def stopwords_eng_fa(self):
        if  not os.path.isfile(os.path.join(self.current_dir, 'models', 'stopwords_eng_fa.txt')):
            with open(os.path.join(self.current_dir, 'models', 'stopwords.txt'), 'r', encoding='utf8') as stopwords_file:
                self.stopwords = stopwords_file.readlines()
            stopwords = [line.replace('\n', '') for line in stopwords] 
            nltk.download('stopwords')
            nltk_stopwords = nltk.corpus.stopwords.words('english')
            stopwords.extend(nltk_stopwords)
            with open(os.path.join(self.current_dir, 'models', 'stopwords_eng_fa.txt'), "w", encoding="utf-8") as f: 
                for stopword in stopwords:
                    f.write(stopword + "\n")
            f.close()
        else:
            model = os.path.join(self.current_dir, 'models', 'stopwords_eng_fa.txt')
            with open(model, encoding='utf8') as stopwords_file:
                stopwords = stopwords_file.readlines()
            stopwords = [str(line).replace('\n', '') for line in stopwords]  
        return(stopwords )
        
    def tokenize_filter_stem_lem (self, news ):
        stemmer = hazm.Stemmer()
        ##lem = hazm.Lemmatizer()  
        txt_tokenized = word_tokenize(news)
        txt_tokenized_filtered = [w for w in txt_tokenized if not w in self.stopwords]
        txt_tokenized_filtered_stemmed = [stemmer.stem(w) for w in txt_tokenized_filtered]
        #txt_tokenized_filtered_lem = [lem.lemmatize(w).replace('#', ' ') for w in txt_tokenized_filtered]
        cleaned_news=' '.join(txt_tokenized_filtered_stemmed) # + ' ' + ' '.join(txt_tokenized_filtered_lem)
        return(cleaned_news)
    
    
    def train_model(self):
        data = pd.read_csv('per.csv')    
        for index, row in data.iterrows():
                    news = " ".join(str(row[x]) for x in ['Title' , 'Body'])   
                    data .loc[index , 'txt'] = self.tokenize_filter_stem_lem (news)
                    data .loc[index , 'category']= row['Category2'].replace('\n', '') 
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2) , encoding='utf8')
        X =self.vectorizer.fit_transform(data['txt'])  
        self.le = LabelEncoder()
        y = self.le.fit_transform(data['category'])
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        if self.model_name == 'XGBoost':
            self.trained_model = xgb.XGBClassifier()
        elif self.model_name == 'KNN':
            self.trained_model=KNeighborsClassifier(n_neighbors =20)
        else: # if model_name == 'SVM':
            self.trained_model = svm.SVC()
        print("Training started....")
        self.trained_model.fit(X_train, y_train) 
        print("Training finished.")
        y_pred = self.trained_model.predict(X_test)
        self.confusion= confusion_matrix(y_test, y_pred)

    def save_results_to_files(self):
        trained_model_model_route  = os.path.join(self.current_dir, 'models', self.model_name+'_model.jdsh')
        trained_model_model_file = open(trained_model_model_route , 'wb')
        pickle.dump(self.trained_model ,trained_model_model_file )
        trained_model_model_file.close()

        trained_model_le_route = os.path.join(self.current_dir, 'models', self.model_name+'_le.jdsh')
        trained_model_le_file = open(trained_model_le_route , 'wb')
        pickle.dump(self.le ,trained_model_le_file )
        trained_model_le_file.close()

        trained_model_vectorizer_route = os.path.join(self.current_dir, 'models', self.model_name+'_vectorizer.jdsh')
        trained_model_vectorizer_file = open(trained_model_vectorizer_route , 'wb')
        pickle.dump(self.vectorizer ,trained_model_vectorizer_file )
        trained_model_vectorizer_file.close()

        trained_model_confusion_route = os.path.join(self.current_dir, 'models', self.model_name+'_confusion.jdsh')
        trained_model_confusion_file = open(trained_model_confusion_route , 'wb')
        pickle.dump(self.vectorizer ,trained_model_confusion_file )
        trained_model_confusion_file.close()

    def read_from_cached_files(self):
        trained_model_model_route  = os.path.join(self.current_dir, 'models', self.model_name+'_model.jdsh')
        if   os.path.isfile(trained_model_model_route):
            trained_model_model_file = open(trained_model_model_route, 'rb') 
            self.trained_model = pickle.load(trained_model_model_file)
            trained_model_model_file.close()    
        else:
            print("NO cached trained model file exists")
            return(False)
        
        trained_model_le_route = os.path.join(self.current_dir, 'models', self.model_name+'_le.jdsh')
        if   os.path.isfile(trained_model_le_route):
            trained_model_le_file = open(trained_model_le_route, 'rb')
            self.le = pickle.load(trained_model_le_file)
            trained_model_le_file.close()
        else:
            print("NO cached label encoder file exists")
            return(False)
             
        trained_model_vectorizer_route = os.path.join(self.current_dir, 'models', self.model_name+'_vectorizer.jdsh')
        if   os.path.isfile(trained_model_vectorizer_route):
            trained_model_vectorizer_file = open(trained_model_vectorizer_route, 'rb')
            self.vectorizer = pickle.load(trained_model_vectorizer_file)
            trained_model_vectorizer_file.close()    
        else:
            print("NO cached vectorizer file exists")
            return(False)
      

        #trained_model_confusion_route = os.path.join(self.current_dir, 'models', self.model_name+'_confusion.jdsh')
        #if   os.path.isfile(trained_model_confusion_route):
        #    with open(trained_model_confusion_route, 'r', encoding='utf8') as trained_model_confusion_file:
        #        self.confusion = pickle.load(trained_model_confusion_file)
        return(True)

    def predict_cluster_news(self , news ): 
        cleaned_news= self.tokenize_filter_stem_lem (news )
        new_entry=[cleaned_news]
        x_v = self.vectorizer.transform(new_entry)
        y_pred =self.trained_model.predict(x_v)
        label = self.le.inverse_transform(y_pred)
        return label[0]

########################################################################################################################
st.set_page_config(layout="wide") 
# Sidebar 
train_model_select_box = st.sidebar.selectbox("Train model", ["SVM", "XGBoost" , 'KNN'])
print(train_model_select_box) 
model_obj = TrainTestModel(train_model_select_box)
# Category prediction page
#st.write(st.session_state)

st.title("Category Prediction")
st.write("Please press 'enter' to submit the URL or press 'ctrl+enter' to submit the text")
st.write()

if 'radio' not in st.session_state:
    st.session_state['radio'] = 'URL'
col1, col2 = st.columns(2)
with col1:    
    category_type_radio_button = st.radio("Category type:", ["URL", "News Text"] , key='radio') 
with col2: 
    url = st.text_input("URL:" , key='txtarea' , disabled=(st.session_state.radio!='URL'))
    news= st.text_area("News Text:" , key='txtbox' , disabled=(st.session_state.radio!='News Text') )

    
predicted_category_txt = ""
if (st.session_state.radio=='News Text') and (news != "") : 
        predicted_category_txt = model_obj.predict_cluster_news(news) 
        st.subheader("Recognized category is:")
        st.success(predicted_category_txt)
else: 
    urlregex = re.compile( r'^(?:http|ftp)s?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)]|'
                            r':|=;?/|,%])+$')
    if urlregex.match(url):
            soup = BeautifulSoup(urlopen(url))
            news = ' '.join(map(lambda p:p.text,soup.find_all('p')))
            predicted_category_txt = model_obj.predict_cluster_news(news )
            st.subheader("Recognized category is:")
            st.success(predicted_category_txt)        
    


# python -m streamlit run temp.py --server.port 8080  