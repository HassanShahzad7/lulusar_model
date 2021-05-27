
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib
import pickle

df = pd.read_excel('Lulusar_data.xlsx')
df.columns = df.columns.str.replace('Data ', '')

df.rename(columns = lambda x: x.replace(' ', '_'), inplace = True)
cols = ['Date_Start', 'Date_Stop', 'Paging_Cursors_Before', 'Paging_Cursors_After', 'Purchase_Roas_Action_Type', 'Optimization_Goal']
df = df.drop(cols, axis = 1)
df.drop_duplicates(subset = ['Purchase_Roas_Value', 'Impressions'],keep = 'first', inplace = True)
a = ['MESSAGES', 'PRODUCT_CATALOG_SALES', 'STORE_VISITS', 'EVENT_RESPONSES', 'REACH', 'PAGE_LIKES']
df = df[~df.Objective.isin(a)]
print(df)

y = df.Objective
x = df.loc[:, df.columns != 'Objective']
x.head()
x.Purchase_Roas_Value = x.Purchase_Roas_Value.fillna(np.mean(x.Purchase_Roas_Value))

le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)
sm = SMOTE(random_state=42)
x_sm, y_sm = sm.fit_resample(x, y)
print(x.shape, y.shape)
print(x_sm.shape, y_sm.shape)
x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size = 0.25, random_state = 42, stratify = y_sm)
loaded_model = pickle.load(open("rf.pickle.dat", "rb"))
loadedd = pickle.load(open("gb.pickle.dat", "rb"))
load_pred = loaded_model.predict(x_test)
load_pred1 = loadedd.predict(x_test)
acc1 = accuracy_score(y_test, load_pred1)*100 
acc = accuracy_score(y_test, load_pred)*100
print(acc)
st.title('Lulusar Objective Prediction')
classifier_name = st.sidebar.selectbox('Select Classifier', ('Random Forest', 'Gradient Boosting'))
#gb_score = accuracy_score(y_test, gb_pred)*100
#rf_score = accuracy_score(y_test, rf_pred)*100

y_pred = loaded_model.predict(x_test)
st.write('Accuracy Gradient Boosting: ' + "{:.2f}".format(acc1))
st.write('Accuracy Random Forest: ' + "{:.2f}".format(acc))
cpic = float(st.sidebar.text_input('Cost Per Inclick: ', 8.5))
freq = float(st.sidebar.text_input('Frequency: ', 4.6))
clicks = int(st.sidebar.text_input('Clicks: ', 1500))
cpc = float(st.sidebar.text_input('CPC: ', 11.5))
cpm = float(st.sidebar.text_input('CPM: ', 23.5))
cpp = float(st.sidebar.text_input('CPP: ', 15.5))
ctr = float(st.sidebar.text_input('CTR: ', 3.5))
imp = int(st.sidebar.text_input('Impressions: ', 4500))
reach = int(st.sidebar.text_input('Reach: ', 3000))
spend = float(st.sidebar.text_input('Spend: ', 3500))
roas = float(st.sidebar.text_input('ROAS: ', 100))
lister = [cpic, freq, clicks, cpc, cpm, cpp, ctr, imp, reach, spend, roas]

if classifier_name == 'Random Forest':
	if loadedd.predict([lister]) == [0]:
		st.write('Objective: Brand Awareness')
	if loadedd.predict([lister]) == [1]:
		st.write('Objective: Conversions')
	if loadedd.predict([lister]) == [2]:
		st.write('Objective: Link Clicks')
	if loadedd.predict([lister]) == [3]:
		st.write('Objective: Post Engagement')
	if loadedd.predict([lister]) == [4]:
		st.write('Objective: Video Views')
else: 
	if loaded_model.predict([lister]) == [0]:
		st.write('Objective: Brand Awareness')
	if loaded_model.predict([lister]) == [1]:
		st.write('Objective: Conversions')
	if loaded_model.predict([lister]) == [2]:
		st.write('Objective: Link Clicks')
	if loaded_model.predict([lister]) == [3]:
		st.write('Objective: Post Engagement')
	if loaded_model.predict([lister]) == [4]:
		st.write('Objective: Video Views')
