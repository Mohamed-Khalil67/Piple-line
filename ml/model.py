from utils.utils import DataHandler,FeatureRecipe,FeatureExtractor

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
