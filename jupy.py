#importer vos librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class DataHandler:
    def __init__(self) :
        self.dataprice = None
        self.datalisting = None
        self.merge = None
    def get_data(self):
        print("_ _ Fetch Data From bucket _ _")
        self.dataprice=pd.read_csv("price_availability.csv",sep=';')  # What is csv : comma separateur virgule
        self.datalisting=pd.read_csv("listings_final.csv",sep=';')
        return "_ _  data loaded _ _\nFiles : \n - listing {} \n - prices {} ".format(self.datalisting.shape,self.dataprice.shape)
    def group_data(self):
        data = self.dataprice.groupby('listing_id')['local_price'].mean()
        self.merge = pd.merge(data, self.datalisting, on='listing_id')
        print(" - - - data merged - - - ")
    def get_process_data(self):
        self.get_data()
        self.group_data()
        print(self.merge)
        print("_ _ data processed _ _")

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

        if "Unnamed : 0" in self.data.columns :
            self.data.drop(['Unnamed: 0'],axis='columns',inplace=True)

        for col in self.data.columns :
            if self.data[col].isna().sum() == len(self.data[col]) :
                self.data.drop([col],axis='columns',inplace=True)

        print("--Done Dropping--")
    
    def deal_duplicate(self):
        """ TODO : Supprimer les lignes dupliquées du dataset """
        
        print("-- Duplicas Deletion commenced --")
        self.data.drop_duplicates(inplace=True)
        print("--Duplicas Deteted --")

    def drop_nanp(self, threshold: float):
        """ 
            TODO : Supprimer un pourcentage de NA du dataset 

            Threshold is inserted from 0 to 100 float variable
        """
        
        dropped=[]
        print("-- threshold est mis en Argument {}-- ".format(threshold))
        for col in self.data.columns :
            if (self.data[col].isna().sum()/self.data.shape[0]) >= threshold/100 : 
                dropped.append(col)
        print("{} feature have more than {} NaN ".format(len(col), threshold))
        print('\n\n - - - features - - -  \n {}'.format(col))    
        
        self.data.drop(dropped,axis='columns',inplace=True)
        print("-- Columns Dropped {}".format(dropped))
    
    def deal_dtime(self):
        """ TODO : Traiter les DateTime """
        pass

    def prepare_data(self, threshold: float):
        self.separate_variable_types()
        self.drop_uselessf()
        self.deal_duplicate()
        self.drop_nanp(threshold)
        self.deal_dtime()    


class FeatureExtractor:
    """
    Feature Extractor class
    """    
    def __init__(self, data: pd.DataFrame, flist: list):
        """
            Input : pandas.DataFrame, feature list to drop
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """