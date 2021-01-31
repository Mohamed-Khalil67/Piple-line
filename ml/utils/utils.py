#importer vos librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

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
        #self.model en plsu de model baf... si j'ai pas chargé alors pas de modéle,s'il y a alors charge
        try:
            #load model
            return joblib.load("{}/model_base.joblib".format(self.model_path))
            ldm = joblib.load("{}/model_base.joblib".format(self.model_path))
            print("File Loaded Successfully")
        except:
            print("File doesn't exist.You must save the model first")