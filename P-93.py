# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})


# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})

# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression(n_jobs = -1,max_iter = 100)
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1,n_estimators=100)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


@st.cache_data()
def prediction(_model,island,bill_length,bill_depth,filler_length,body_mass,sex) :
  species = _model.predict([[island,bill_length,bill_depth,filler_length,body_mass,sex]])
  species = species[0]
  st.write("The prediced species are :", species)
  st.write("0 - Adelie\n1-Chinstrap\n2-Gentoo")
  if species == 0:
    return 'Adelie'
  elif species == 1:
    return 'Chinstrap'
  else :
    return 'Gentoo'
  


st.title('Pengiun prediction')
bill_length = st.sidebar.slider('bill_length_mm',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bill_depth = st.sidebar.slider('bill_depth_mm',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
flipper_length = st.sidebar.slider('flipper_length_mm',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
body_mass = st.sidebar.slider('body_mass_g',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))


island = st.sidebar.selectbox("island",("Biscoe",'Dream','Torgersen'))
if island == 'Biscoe':
  island = 0
elif island == 'Dream' :
  island = 1
else :
  island = 2


sex = st.sidebar.selectbox("sex",('Male','Female'))
if sex == 'Male' :
  sex = 0
else :
  sex = 1

classifier = st.sidebar.selectbox('Classifier',('Support vector machine','LogisticRegression','RandomForestClassifier'))

if st.sidebar.button('Predict') :

  if classifier == 'Support vector machine' :
    species_class = prediction(svc_model,island,bill_length,bill_depth,flipper_length,body_mass,sex)
    species_score = svc_model.score(X_train,y_train)
    st.write("The selected bill_length : ",bill_length)
    st.write("The selected bill_depth : ",bill_depth)
    st.write("The selected flipper_length : ",flipper_length)
    st.write("The selected body_mass : ",body_mass)
    st.write("The score of support vector machine : ",species_score)

  elif classifier == 'LogisticRegression' :
    species_class = prediction(log_reg,island,bill_length,bill_depth,flipper_length,body_mass,sex)
    species_score = log_reg.score(X_train,y_train)
    st.write("The selected bill_length : ",bill_length)
    st.write("The selected bill_depth : ",bill_depth)
    st.write("The selected flipper_length : ",flipper_length)
    st.write("The selected body_mass : ",body_mass)
    st.write("The score of LogisticRegression : ",species_score)

  else :
    species_class = prediction(rf_clf,island,bill_length,bill_depth,flipper_length,body_mass,sex)
    species_score = rf_clf.score(X_train,y_train)
    st.write("The selected bill_length : ",bill_length)
    st.write("The selected bill_depth : ",bill_depth)
    st.write("The selected flipper_length : ",flipper_length)
    st.write("The selected body_mass : ",body_mass)
    st.write("The score of RandomForestClassifier : ",species_score)

    

