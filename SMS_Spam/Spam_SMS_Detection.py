import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df=pd.read_csv('spam.csv', encoding = 'latin')
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df['v1'].replace({'ham':0,'spam':1},inplace=True)
print(df.info())
px.pie(data_frame=df,names='v1')
X=df['v2']
y=df['v1']
vec=TfidfVectorizer()
vectorized_X=vec.fit_transform(X)
MLP_pipe = Pipeline([ ('Classifier', MLPClassifier(activation='tanh',hidden_layer_sizes=(100,)))])
X_train, X_test, y_train, y_test =train_test_split(vectorized_X,y,random_state=42,stratify=y,test_size=0.3)
MLP_pipe.fit(X_train,y_train)
pred=MLP_pipe.predict(X_test)
print(classification_report(y_test,pred))