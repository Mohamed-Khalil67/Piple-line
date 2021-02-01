# Project Introduction : Pipeline Data GCP " Cloud Computing "

## Setps For solving a data analyzing and Cleaning in an elborated Way

- Collecting Data

- API Project Architecture Tree 

- API developpement 

- Docker Packaging 

- API Streamlit Link 

## Collecting Data :

### Data analyzing and cleaning is done here by multiple Steps :-

Classification of the file utils.py

1 - Data Handling : 
```
class DataHandler:
    def __init__(self) :
        self.dataprice = None
        self.datalisting = None
        self.merge = None
    def get_data(self):
        print("_ _ Fetch Data From bucket _ _")
        self.dataprice=pd.read_csv("https://storage.googleapis.com/h3-data/price_availability.csv",sep=';')  # What is csv : comma separateur virgule
        self.datalisting=pd.read_csv("https://storage.googleapis.com/h3-data/listings_final.csv",sep=';')
        return "_ _  data loaded _ _\nFiles : \n - listing {} \n - prices {} ".format(self.datalisting.shape,self.dataprice.shape)
    def group_data(self):
        print(" - - - Start Data Merging - - - ")
        data = self.dataprice.groupby('listing_id')['local_price'].mean()
        self.merge = pd.merge(data, self.datalisting, on='listing_id')
        print(" - - - data merged - - - ")
    def get_process_data(self):
        print(" - - - Start Data Processing - - - ")
        self.get_data()
        self.group_data()
        return self.merge
        print("-- End of DataHandling --") 
        

```

2 - Feature Recipe :

```
class FeatureRecipe:

    def __init__(self,data:pd.DataFrame):
        """ TODO : __init__ """

        self.data = data
        self.continuous = None # float
        self.categorical = None # Object
        self.discrete = None # int
        self.other = None
        self.datetime = None

    def separate_variable_types(self) -> None:
        """" TODO : Diviser les types de variables dans un tableau """
        print("-- Diviser les types de variables dans un tableau --")
        self.continuous = [] # float
        self.categorical = [] # Object
        self.discrete = [] # int   
        self.other = [] # other ( bool )
        for col in self.data.columns :
            if self.data[col].dtypes == int :
                self.discrete.append(self.data[col])
            elif self.data[col].dtypes == object :
                self.categorical.append(self.data[col])
            elif self.data[col].dtypes == float :
                self.continuous.append(self.data[col])
            else:
                self.other.append(self.data[col])
                          
        print(" The Amount of sepereted types:\n Categories {} \n Discrete {} \n Continuous {} \n Other {}".format(len(self.categorical),len(self.discrete),len(self.continuous),len(self.other)))
    
    def drop_uselessf(self):
        """ TODO : Supprimer les colonnes inutiles du dataset """
        print("-- Dropping uselese Columns--")

        if "Unnamed: 0" in self.data.columns :
            self.data.drop(['Unnamed: 0'],axis='columns',inplace=True)

        for col in self.data.columns :
            if self.data[col].isna().sum() == len(self.data[col]) :
                self.data.drop([col],axis='columns',inplace=True)

        print("--Done Dropping--")
    
    def get_duplicates(self):
        duplicates = []
        for col in range(self.data.shape[1]):
            for scol in range(col+1,self.data.shape[1]):
                if self.data.iloc[:,col].equals(self.data.iloc[:,scol]):
                    duplicates.append(self.data.iloc[:,scol].name)
        return duplicates
    
    def drop_duplicate(self):
        # comparer les colonnes et voir si elles sont dupliquées
        print("dropping duplicated rows")
        self.data.drop_duplicates(inplace=True)
        duplicates = self.get_duplicates()
        for col in duplicates:
            print("dropping column :{}".format(col))
            self.data.drop([col], axis='columns', inplace=True)
        print("duplicated rows dropped")

    def drop_nanp(self, threshold: float):
        """ 
            TODO : Supprimer un pourcentage de NA du dataset 

            Threshold is inserted from 0 to 100 float variable
        """
        def deal_nanp(data:pd.DataFrame, threshold: float):
            bf=[]
            for c in self.data.columns.to_list():
                if self.data[c].isna().sum()/self.data.shape[0] >= threshold:
                    bf.append(c)
            print("{} feature have more than {} NaN ".format(len(bf), threshold))
            print('\n\n - - - features - - -  \n {}'.format(bf))
            return bf 
        
        self.data = self.data.drop(deal_nanp(self.data, threshold), axis=1)
        print('Some NaN features droped according to {} thresold'.format(threshold))
    
    def deal_dtime(self):
        """ TODO : Traiter les DateTime """
        pass

    def prepare_data(self, threshold: float):
        print(" - - - Start Preparing Data - - - ")
        self.drop_uselessf()
        self.drop_duplicate()
        self.drop_nanp(threshold)
        self.separate_variable_types()
        self.deal_dtime()   
        print("-- End of FeatureRecipe --") 

```

3 - Feature Extractor : 

```
class FeatureExtractor:
    """
    Feature Extractor class
    """    
    def __init__(self, data: pd.DataFrame, flist: list):
        """
            Input : pandas.DataFrame, feature list to drop
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
        self.df = data
        self.flist = flist
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def extract(self):
        print("-- X and y Values to be put while discarding the flist that we dont need--")
        for col in self.flist:
            if col in self.df:
                self.df.drop(col,axis="columns",inplace=True)

        print("-- Extraction Done --")

    def split(self,size:float,rand:int,y:str):
        print("-- Using Split mehtode --")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.loc[:,self.df.columns!=y], self.df[y], test_size=size, random_state=rand)
        print("-- Afficher la shape de vos données --")

    def get_process_data(self,size:float,rand:int,y:str):
        self.extract()
        self.split(size,rand,y)
        return self.X_train, self.X_test, self.y_train, self.y_test
        print("-- Done processing Feature Extractor --")


```

4 - ModelBuilder : 

```
class ModelBuilder:
    """
        Class for train and print results of ml model 
    """
    def __init__(self, model_path: str = None, save: bool = False):
        self.model_path = model_path
        self.save = save
        self.reg = LinearRegression()
        #self.time = DT.datetime.now()

    def __repr__(self): # class courier python , affichage par default isntancié , methode __str__ 
        return f'Model Builder : model_path = {self.model_path} , save = {self.save}.'

    def train(self,X,y):
        print(" - - - Training Start - - - ")
        self.reg.fit(X,y)
        print(" - - - Finish training - - - ")

    def predict_test(self, X) -> np.ndarray:  # certain ligne of X_test or y_test
        print(" - - - Tesitng Certain Ligne - - - ")
        self.reg.predict(X[0])        
        print(" - - - Finish Certain Ligne - - - ")

    def predict_from_dump(self, X) -> np.ndarray: #
        print(" - - - Tesitng From Dump - - - ")
        self.reg.predict(X)
        print(" - - - Finished Testing From Dump - - - ")

    def save_model(self): # joblib saving de predict_test of fit , et on faire le dump de joblib par la focntion de predict_from_dump
        #with the format : 'model_{}_{}'.format(date)
        print(" - - - Saving Model - - - ")
        joblib.dump(self.reg,"{}/model.joblib".format(self.model_path))
        print(" - - - Finished Saving Model - - - ")

    def print_accuracy(self,X_test,y_test): # accuracy c'est le score 
        print(" - - - Accuracy Printing - - - ")
        self.reg.score(X_test,y_test)
        print("Coeffecient Accuracy : {} %".format(self.reg.score(X_test,y_test)*100))
        print(" - - - Finished Accuracy - - - ")

    def load_model(self): # à partir de fichier joblib , je le mets en instance 
```

5 - DataManager : 

```
def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None): # script model.py qui sort une model .joblib , alors dans model.py datamanager model builder et va sortir un fichier .joblib
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)

    Args:
        d (DataHandler, optional): [description]. Defaults to None
        fr (FeatureRecipe, optional): [description]. Defaults to None
        fe (FeatureExtractor, optional): [description]. Defaults to None
    """
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

    return X_train,X_test,y_train,y_test

```

### API Architecture Tree :-

ml_template_api

```
 ├── main.py
 ├── requirements.txt
 ├── Dockerfile
 └── ml
     ├── __init__.py
     │
     ├── model.joblib
     │
     ├── EDA.ipynb
     │
     ├── algo_pipeline_demo.ipynb
     │
     └── model.py
     │
     └── utils
         ├── __init__.py
         └── utils.py
         
```
         
  
 ### API developpement :-
 
```
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
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
```

 ### Docker Packaging :-
 
 - Docker File :
```
# Version Python alpine est plus petit et pyhton 3.6 
FROM python:3.8-slim 
COPY . /app
WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install streamlit
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]

```
- docker-compose.yml :

```
version: '3'

services:
    app:
        image: streamlit-docker
        build:
            dockerfile: ./Dockerfile
            context: .
        environment:
            - PORT=8000
        ports:
            - ${PORT}:${PORT}
        volumes:
            - ./app:/usr/app/src/
```
 
 ### API Streamlit Link :-
 
 - VM instance est crée avec une activation d'api streamlit de port 8000
 
    -- Streamlit Link : http://104.199.46.74:8000/
    terminal commande : Streamlit run main.py --server.port=8000 
 - VM instance jupyter port connexion à 8888 : 
 
    -- 
      



