# Dans cette feuille on va faire un main pour faire tourner nos models sur google cloud : 
# Pour faire le test il faut que je modifie ma fonction afin de faire le test pour chaque id
import os
from flask import Flask, request, jsonify
from datetime import datetime
from typing import Union
# Import specific functions or classes from your other files
from utils import charge_from_google_Drive
from utils import create_features
from utils import load_registered_mlforecast
from utils import split_df 
from utils import add_ds_unique_id
from OptiModels import create_and_use_optuna
import FunctionWineChargeUniqueTFT
import TFTModelErrorExplanation
import shap

app = Flask(__name__)

#Fonction de test 
@app.route('/test', methods=['GET'])
def test():
    return "Test route working"

#Fonction d'entrainement
@app.route('/trainModel', methods=['POST'])
def trainModelForTest() :  
    
    Product_Id_produit_json = request.json['ID_PRODUIT']
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./optunasavedoncloud-715a0f78dedf.json"
    os.path.isfile("./optunasavedoncloud-715a0f78dedf.json")

    file_path = "https://storage.googleapis.com/wine_product_data_frame/WineProductDataFrame.csv"
    df_final = charge_from_google_Drive(file_path)
    
    
    print(Product_Id_produit_json)
    
    # print(df_final)
    

    model_forecast = 'xgboost'
    target = 'QUANTITE'
    nomencalture = 'Vin'
    id_produit = int(Product_Id_produit_json)
    current_date = datetime.now()
    DATE_UPDATE_optuna = current_date.strftime('%d/%m/%Y')
    exogenous = [x for x in df_final.columns if x != 'QUANTITE']
    df_produit =  df_final[df_final['ID_PRODUIT'] == id_produit]
    print(df_produit)

    model, best_params, study_XGBoost, df_final_train, df_final_test = create_and_use_optuna(df_produit, DATE_UPDATE_optuna, nomencalture, model_forecast, target, exogenous, id_produit)
    
    return jsonify({"message": "Model training completed", "product_id": id_produit}), 200

#Fonction pour contrôler si le modèle se télécharge bien
@app.route('/getModel', methods=['GET'])
def getModel():
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./optunasavedoncloud-715a0f78dedf.json"
    os.path.isfile("./optunasavedoncloud-715a0f78dedf.json")
    id_produit = request.json['ID_PRODUIT']
    if not id_produit:
        return jsonify({"error": "Please provide a product ID"}), 400
    
    try:
        id_produit = int(id_produit)
    except ValueError:
        return jsonify({"error": "Invalid product ID format"}), 400
    
    try:
        file_path = "https://storage.googleapis.com/wine_product_data_frame/WineProductDataFrame.csv"
        df_final = charge_from_google_Drive(file_path)

        model_forecast = 'xgboost'
        target = 'QUANTITE'
        nomencalture = 'Vin'
        id_produit = int(id_produit)
        current_date = datetime.now()
        DATE_UPDATE_optuna = current_date.strftime('%d/%m/%Y')
        exogenous = [x for x in df_final.columns if x != 'QUANTITE']
        df_produit =  df_final[df_final['ID_PRODUIT'] == id_produit]
        bucket_name = "wine_product_data_frame"
        destination_blob_name = f"Xgboost{id_produit}Model_MLForecast_test"
        
        df_produit = add_ds_unique_id(df_produit) 
        df_final_train, df_final_test = split_df(df_produit,90)
        
        loaded_forecast = load_registered_mlforecast(destination_blob_name)
        print(loaded_forecast)
        # loaded_forecast.feature_names_in_
        exogenous_MLForecast = [x for x in df_final_train.columns if x not in ['QUANTITE']]
        
        prediction = loaded_forecast.predict(
                        h=90,
                        X_df = df_final_test[exogenous_MLForecast]
                        )
        print(prediction)
        
        loaded_forecast
        
        if loaded_forecast is None:
            return jsonify({"error": "Failed to load the model"}), 500

        # Reste de votre code pour traiter le modèle chargé...
        
        return jsonify({
            "message": "Model loaded successfully",
            "product_id": id_produit,
            # Ajoutez ici d'autres informations pertinentes sur le modèle
        }), 200
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            "error": f"Error loading model: {str(e)}",
            "details": error_details
        }), 500
        
        
#Function to save the result obtained from a one to one boosting 
# @app.route('/getCausalsExplanations/XGBoost', methods=['POST'])
# def SaveModelandResultXGBoost() : 
    
#     return


#Function to save the result obtained using a one to one tft 
@app.route('/getCausalsExplanations/TFT', methods=['POST','GET'])
def SaveModelandResultTFT() : 
    nb_epoch = 20
    n_days = 90 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./optunasavedoncloud-715a0f78dedf.json"
    os.path.isfile("./optunasavedoncloud-715a0f78dedf.json")
    id_produit = request.json['ID_PRODUIT']
    df_final_reduced = FunctionWineChargeUniqueTFT.ChargeDataBase_(id_produit)
    df_final_train_singleProduct, df_final_test_singleProduct, df_final_reduced_singleProduct = FunctionWineChargeUniqueTFT.select_product(id_produit, df_final_reduced,n_days)
    TFTModelErrorExplanation.main_training_TFTModelProduitII(nb_epoch, id_produit,df_final_reduced)
    

    if not id_produit:
        return jsonify({"error": "Please provide a product ID"}), 400
        
    return jsonify({"message": "Model training completed", "product_id": id_produit}), 200


    
if __name__ == '__main__':
    # In the app section directly

    # app.debug = False

    app.run(debug= True, host='0.0.0.0', port=8080)

