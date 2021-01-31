import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
#import os
from ml.utils.utils import DataHandler,FeatureRecipe,FeatureExtractor,ModelBuilder

st.header(' Pipeline Data GCP " Cloud Computing " ')
st.markdown(" Two File Data CSV : listings_final.csv / price_availability.csv ")

print("Start Data Managing")
d = DataHandler()
d.get_process_data()
print("Data Loaded")
fr = FeatureRecipe(d.merge)
print("Filtering data with threshold de 40%")
fr.prepare_data(0.3)
print("Filtering done")

print("Feature Extracting ( Split )")
flist = ['listing_id','city','neighborhood','latitude','longitude','is_rebookable','is_new_listing','is_fully_refundable','is_host_highly_rated','is_business_travel_ready','pricing_weekly_factor','pricing_monthly_factor','name','type'] 
print("Flist has been Chosen")
fe = FeatureExtractor(d.merge,flist)
X_train,X_test,y_train,y_test = fe.get_process_data(0.3,42,'local_price')

m = ModelBuilder('.') # model_path directory
m.train(X_train, y_train)
m.print_accuracy(X_train,y_train)

#a = "https://storage.googleapis.com/h3-data/listings_final.csv"
#b = "https://storage.googleapis.com/h3-data/price_availability.csv"

#base_path = [a,b]

#list_dataset = os.listdir(base_path)
#list_dataset = [a,b]
#options = list(range(len(list_dataset)))
#path = st.selectbox("files.csv", options, format_func=lambda x: list_dataset[x])  

#df = pd.read_csv(list_dataset[path],sep=';')

df = d.merge

st.title(" Airbnb Web Application")
st.markdown("Data Cleaning And Presentation of Airbnb")
st.header(" Airbnb Pricing Service Algorithme ")

if st.checkbox("Show DataSet Rows of header"):
    number = st.number_input("Number of Rows to View")
    st.dataframe(df.head(int(number)))

selec = []
for col in df.columns:
   selec.append(col)
option = st.multiselect("Selectionner les colonnes",(selec))

st.table(df[option].head())

st.header("Discription Choices Boxes :")

if st.button("Afficher le type des colonnes du dataset"):
    st.table(df[option].dtypes)
if st.button("La shape du dataset, par lignes et par colonnes"):
    st.write(df[option].shape)
if st.button("Afficher les statistiques descriptives du dataset"):
    st.write(df[option].describe())

st.header('Split Details :-')
st.subheader('X_train :')
st.write(X_train)
st.subheader('X_test :')
st.write(X_test)
st.subheader('y_train :')
st.write(y_train)
st.subheader('y_test :')
st.write(y_test)

st.subheader('Split Accuracy :')
st.write('Coefficient Accuracy : ',m.reg.score(X_test,y_test)*100,'%')

#graph = []
#graph = st.multiselect("What kind of graph ?",("Boxplot","Correlation"))
#st.write("You selected",len(graph),"graph")
#if 'Boxplot' in graph:
#    df[option].boxplot()
#    st.title("Distribution des Variables d'intérêt")
#    st.pyplot()
#elif 'Correlation' in graph:
#    st.write(sns.heatmap(df[option].corr(), annot=True))
#    st.pyplot()

st.header("What kind of graph ?")
status = st.radio("Choose ur way of drawing :",('Boxplot','Correlation','Distplot','Pairplot','Histogram'))
if status == 'Histogram':
    fig=plt.gcf()
    fig.set_size_inches(20,5)
    st.write(plt.hist(df.local_price, bins = 25))
    st.pyplot()
elif status == 'Boxplot':
    fig=plt.gcf()
    fig.set_size_inches(20,5)
    st.write(sns.boxplot(x='beds',y='local_price',data=df))
    st.pyplot()
elif status == 'Correlation':
    st.write(sns.heatmap(df.corr(), annot=True))
    st.pyplot()
elif status == 'Distplot':
    st.write(sns.distplot(df))
    st.pyplot()
elif status == 'Pairplot':
    st.write(sns.pairplot(df))
    st.pyplot()