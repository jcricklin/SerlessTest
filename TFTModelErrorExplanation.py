#On ecrit ici l'ensemble des fonctions nécessaire au calcule des modèles : 
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from io import BytesIO
import shutil
import tempfile
import os
import pickle

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from oauth2client.client import GoogleCredentials
import pickle
import tempfile
from google.cloud import storage

import time
from datetime import datetime

import torch


from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import StaticCovariatesTransformer, MissingValuesFiller
from darts.explainability import TFTExplainer

from scipy import stats


#On fait une fonction prépare les données pour Darts :
def PrepareDataForDarts(df_final_test_singleProduct, df_final_train_singleProduct, df_final_reduced_single_product) :
    TIME_COL = "ds"
    TARGET = "QUANTITE"
    STATIC_COLS = ["unique_id"]
    FREQ = "D"
    FORECAST_HORIZON = df_final_test_singleProduct[TIME_COL].nunique()
    COVARIATES = [x for x in df_final_test_singleProduct.columns if x not in [TIME_COL, TARGET] + STATIC_COLS]
    SCALER = Scaler()
    TRANSFORMER = StaticCovariatesTransformer()
    PIPELINE = Pipeline([SCALER, TRANSFORMER])

    total_darts = TimeSeries.from_group_dataframe(df=df_final_reduced_single_product,
                                                group_cols=STATIC_COLS,
                                                time_col=TIME_COL,
                                                value_cols=TARGET,
                                                freq=FREQ,
                                                fill_missing_dates=True,
                                                fillna_value=0)

    # read train and test datasets and transform train dataset
    train_darts = TimeSeries.from_group_dataframe(df=df_final_train_singleProduct,
                                                group_cols=STATIC_COLS,
                                                time_col=TIME_COL,
                                                value_cols=TARGET,
                                                freq=FREQ,
                                                fill_missing_dates=True,
                                                fillna_value=0)
    test_darts = TimeSeries.from_group_dataframe(df=df_final_test_singleProduct,
                                                group_cols=STATIC_COLS,
                                                time_col=TIME_COL,
                                                value_cols=TARGET,
                                                freq=FREQ,
                                                fill_missing_dates=True,
                                                fillna_value=0)

    create_covariates = []
    for ts in tqdm(total_darts):

        unique_id = ts.static_covariates['unique_id'].item()
        # TVA = ts.static_covariates['TVA'].item()

        # Create covariates to fill with interpolation
        covariate = TimeSeries.from_dataframe(
            df_final_reduced_single_product[df_final_reduced_single_product['unique_id'] == unique_id][[x for x in df_final_reduced_single_product.columns if x != 'QUANTITE']],
            time_col=TIME_COL,
            value_cols=COVARIATES,
            freq=FREQ,
            fill_missing_dates=True
        )

        # Align other_cov to match the time axis of the main series
        other_cov = TimeSeries.from_dataframe(df_final_reduced_single_product[(df_final_reduced_single_product['unique_id'] == unique_id)], time_col=TIME_COL, value_cols=STATIC_COLS, freq=FREQ, fill_missing_dates=True)

        # # Stack the aligned other_cov to covariate
        covariate = covariate.stack(other_cov)

        create_covariates.append(covariate)
    covariates_transformed = SCALER.fit_transform(create_covariates)
    train_transformed = PIPELINE.fit_transform(train_darts)

    return PIPELINE, train_transformed, covariates_transformed, create_covariates, train_darts, test_darts, total_darts


# Upload my file
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


#Training version of my one to one TFT
def TrainSaveDartsTFT(train_transformed, covariates_transformed, nb_epoch, num_id) :
    # Your existing setup
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./optunasavedoncloud-715a0f78dedf.json"
    os.path.isfile("./optunasavedoncloud-715a0f78dedf.json")

    current_date = datetime.now()
    DATE_UPDATE_optuna = current_date.strftime('%d%m%Y')

    # Spécifie les chemins corrects pour sauvegarder le modèle et les checkpoints
    gcs_bucket_name = "wine_product_data_frame"
    gcs_model_dir = f"models/tft_darts_model_{num_id}_{nb_epoch}"
    gcs_checkpoint_dir = f"tft_checkpointsII_{num_id}_{DATE_UPDATE_optuna}"


    # Create a temporary directory for local checkpoints
    temp_dir = tempfile.mkdtemp()

    working_path = os.path.join(temp_dir, gcs_checkpoint_dir)
    # TFT_params without any GCS-related stuff

    TFT_params = {
        # Time serie forecast size :
        "input_chunk_length": 30,
        "output_chunk_length": 15,

        # Saving time series parameters
        "model_name": f"tft_darts_model_{num_id}",
        "work_dir": working_path,
        "save_checkpoints": True,

        # time series hyperparameters
        "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),

        # Hyperparameters for model architecture
        "use_static_covariates": True,
        "hidden_size": 32,
        "lstm_layers": 4,
        "num_attention_heads": 4,

        # Hyperparameters for optimization
        "dropout": 0.1,
        "batch_size": 32,
        "random_state": 42,
        "optimizer_kwargs": {"lr": 1e-3},

        #Nomber of training
        "n_epochs": nb_epoch,
    }

    # Create and train the model
    tft_model = TFTModel(**TFT_params, force_reset=False)
    tft_model = tft_model.fit(
        train_transformed,
        future_covariates= covariates_transformed,
        verbose=True
    )

    test_path = os.path.join(working_path, f"tft_darts_model_{num_id}","checkpoints")
    # Upload checkpoints to GCS
    for filename in os.listdir(test_path):
        if filename.endswith(".ckpt"):
            local_path = os.path.join(test_path,filename)
            gcs_path = f"{gcs_checkpoint_dir}/checkpoints/{filename}"
            print(f"Uploading {local_path} to gs://{gcs_bucket_name}/{gcs_path}")
            upload_to_gcs(gcs_bucket_name, local_path, gcs_path)
            print(f"Final model saved to gs://{gcs_bucket_name}/{gcs_path}")

    # Save the final model to a temporary file
    temp_model_path = os.path.join(temp_dir, gcs_checkpoint_dir,"_model.pth.tar")
    tft_model.save(temp_model_path)

    # Upload the final model to GCS
    final_model_gcs_path = f"{gcs_checkpoint_dir}/_model.pth.tar"
    upload_to_gcs(gcs_bucket_name, temp_model_path, final_model_gcs_path)

    print(f"Final model saved to gs://{gcs_bucket_name}/{final_model_gcs_path}")

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    return tft_model, temp_dir, gcs_bucket_name, gcs_model_dir, gcs_checkpoint_dir


### Function pour télécharger le blob
def download_blobs(bucket_name, model_name, prefix, temp_dir):


    # Initialize the storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Download each blob
    for blob in blobs :
        # Create the full path for the file

        file_path = os.path.join(temp_dir, prefix, model_name, blob.name.replace(prefix + '/', ''))

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download the blob to the file
        blob.download_to_filename(file_path)


        print(f"Downloaded {blob.name} to {file_path}")
    return temp_dir




### fonction pour le calcul de l'erreur : ###
def mean_absolute_scaled_error_darts(actual, predicted, training):
    n = len(actual)
    d = np.abs(np.diff(training)).sum() / (n - 1)
    errors = np.abs(np.array(actual) - np.array([x[0] for x in predicted]))
    return errors.mean() / d

def mean_absolute_error_darts(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array([x[0] for x in predicted])))

def model_aware_mae_darts(actual, predicted, training):
    mode = stats.mode(training).mode.item()
    mode_freq = (training == mode).mean()

    def model_aware_weight(x):
        return 1 if x != mode else (1 - mode_freq) / mode_freq

    weights = np.array([model_aware_weight(x) for x in actual])
    errors = np.abs(np.array(actual) - np.array([x[0] for x in predicted]))
    return np.average(errors, weights=weights)

def mean_absolute_scaled_error_tft_darts(y_true, y_pred, y_train, ids, ds='ds', unique_id='unique_id', QUANTITE='QUANTITE', model='TFT', for_index = 'tft_local' ):
    results = []

    for id_val in ids:
        # print(y_pred.static_covariates["unique_id"])
        pred = [pred for pred in y_pred if int(pred.static_covariates["unique_id"]) == id_val]
        actual = y_true[y_true["unique_id"] == id_val]['QUANTITE']
        training = y_train[y_train["unique_id"] == id_val]['QUANTITE']


        mase = mean_absolute_scaled_error_darts(actual, pred[0].values(), training)
        mae = mean_absolute_error_darts(actual, pred[0].values())
        aware_mode_mae = model_aware_mae_darts(actual, pred[0].values(), training)

        # Store results for this id_produit
        results.append({
            'unique_id': f'{id_val}_{for_index}', # This assumes unique_id is the same as id_produit
            'mase': mase,
            'mae': mae,
            'awareModeMae': aware_mode_mae
        })

    print(results)

    return pd.DataFrame(results)



### Function to make lag analysis ###
def calcule_rolling_min_lag(y, df_for_output)  :
    rolling_min = df_for_output['importance'][y]
    for x in df_for_output.index :
      if type(x) == int and x < y :
        if rolling_min > df_for_output['importance'][x] :
            rolling_min = df_for_output['importance'][x]
    return rolling_min

def calcule_rolling_max_lag(y,df_for_output)  :
    rolling_max = 0
    for x in df_for_output.index :
      if type(x) == int and x < y :
        if rolling_max < df_for_output['importance'][x] :
            rolling_max = df_for_output['importance'][x]
    return rolling_max

def calcule_rolling_mean_lag(y,df_for_output)  :
    rolling_mean = 0
    for x in df_for_output.index:
      if type(x) == int and x < y :
        rolling_mean += df_for_output['importance'][x]
    rolling_mean_lag = rolling_mean/y
    return rolling_mean_lag

def prepar_for_output(df_for_output) :
    new_index_entries = {}

    for x in df_for_output.index:
        if isinstance(x, int):
            if x in [1, 7, 15, 30]:
                new_index_entries[f'lag{x}'] = df_for_output['importance'][x]

            if x == 7:
                new_index_entries[f'rolling_mean_lag1_window_size{x}'] = calcule_rolling_mean_lag(x,df_for_output)
                new_index_entries[f'rolling_max_lag1_window_size{x}'] = calcule_rolling_max_lag(x,df_for_output)
                new_index_entries[f'rolling_min_lag1_window_size{x}'] = calcule_rolling_min_lag(x,df_for_output)

            if x == 15:
                new_index_entries[f'rolling_mean_lag2_window_size{x}'] = calcule_rolling_mean_lag(x,df_for_output)
                new_index_entries[f'rolling_max_lag2_window_size{x}'] = calcule_rolling_max_lag(x,df_for_output)
                new_index_entries[f'rolling_min_lag2_window_size{x}'] = calcule_rolling_min_lag(x,df_for_output)

            if x == 30:
                new_index_entries[f'rolling_mean_lag3_window_size{x}'] = calcule_rolling_mean_lag(x,df_for_output)
                new_index_entries[f'rolling_max_lag3_window_size{x}'] = calcule_rolling_max_lag(x,df_for_output)
                new_index_entries[f'rolling_min_lag3_window_size{x}'] = calcule_rolling_min_lag(x,df_for_output)

    # Create a new DataFrame with the new index entries
    new_rows = pd.DataFrame(new_index_entries, index=['importance']).T

    # Concatenate the new rows to the existing DataFrame
    df_for_output = pd.concat([df_for_output, new_rows])
    return df_for_output


def tft_Explanation(num_id, modelTFTload, train_transformed, train_darts, create_covariates) :
    total_dico_impTFT_cluster = []
    for id_produit in tqdm([num_id], desc="Processing products"):
        explainer = TFTExplainer(
            modelTFTload,
            background_series= train_transformed[[i for i in range(len(train_darts)) if int(train_darts[i].static_covariates["unique_id"]) == id_produit][0]],
            background_future_covariates= create_covariates[[i for i in range(len(train_darts)) if int(train_darts[i].static_covariates["unique_id"]) == id_produit][0]])
        explainability_result = explainer.explain()

        data_encoder = explainer._encoder_importance
        data_decoder = explainer._decoder_importance
        data = (data_encoder + data_decoder)/200
        somme_attention = sum([sum(explainability_result.get_attention()[y].values()[0]) for y in range(30)])

        for y in range(len(explainability_result.get_attention())) :
                data[y] = sum(explainability_result.get_attention()[y].values()[0])/somme_attention

        dict_transit = [ {'variable': element, 'importance': data[element][0]} for element in data]
        df_for_output = pd.DataFrame(dict_transit, columns=['variable','importance'])
        df_for_output = df_for_output.set_index('variable')
        # print(df_for_output)
        df_id = prepar_for_output(df_for_output)

        importance_dict = {
        'unique_id': f'{int(id_produit)}_tft_produit',
        **dict(zip(df_id.index, df_id['importance']))
        }
        total_dico_impTFT_cluster.append(importance_dict)

    # Create the final DataFrame
    final_feature_importance_df_tft_cluster = pd.DataFrame(total_dico_impTFT_cluster)

    # Set 'unique_id' as the index
    final_feature_importance_df_tft_cluster.set_index('unique_id', inplace=True)

    return final_feature_importance_df_tft_cluster





def function_result_final(nb_epoch, num_id, df_final_test_singleProduct, df_final_train_singleProduct, df_final_reduced_single_product) :
    PIPELINE, train_transformed, covariates_transformed, create_covariates, train_darts, test_darts, total_darts = PrepareDataForDarts(df_final_test_singleProduct, df_final_train_singleProduct, df_final_reduced_single_product)
    tft_model, temp_dir, gcs_bucket_name, gcs_model_dir, gcs_checkpoint_dir = TrainSaveDartsTFT(train_transformed, covariates_transformed, nb_epoch, num_id)

    test = tft_model.predict(n=90,
                            series=train_transformed, # The training periods
                            num_samples=90,
                            future_covariates=covariates_transformed # The entire periods
                            )

    prediction = PIPELINE.inverse_transform(test)

    errorMetric = mean_absolute_scaled_error_tft_darts(df_final_test_singleProduct[df_final_test_singleProduct['ds'] >= df_final_test_singleProduct['ds'].max() - pd.Timedelta(days=89)], prediction, df_final_train_singleProduct, [num_id], for_index = 'tft_produit')

    final_feature_importance_df_tft_cluster = tft_Explanation(num_id, tft_model, train_transformed, train_darts, create_covariates)

    tft_produit_result = final_feature_importance_df_tft_cluster.merge(errorMetric, on = 'unique_id', how = 'left')

    return tft_produit_result

    
def save_csv_to_gcs(destination_blob_name, dataframe):
    # Set the path to your Google Cloud credentials JSON file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./optunasavedoncloud-715a0f78dedf.json"

    # Verify that the credentials file exists
    if not os.path.isfile("./optunasavedoncloud-715a0f78dedf.json"):
        raise FileNotFoundError("Google Cloud credentials file not found.")

    # 1. Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.csv') as temp_file:
        # 2. Write the DataFrame to the temporary CSV file
        dataframe.to_csv(temp_file.name, index=True)
        temp_file_name = temp_file.name

    # 3. Initialize the GCS client
    client = storage.Client()

    # 4. Specify the bucket name
    bucket_name = "wine_product_data_frame"

    # 5. Get a reference to the bucket
    bucket = client.bucket(bucket_name)

    # 6. Create a new blob object and upload the file
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(temp_file_name)

    print(f"File {temp_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")

    # 7. Remove the temporary file
    os.unlink(temp_file_name)

    return temp_file_name, destination_blob_name, bucket_name


#Dans cette version on fait un essai plus proche des besoins de colab
def main_training_TFTModelProduitII(nb_epoch, num_id,df_final_reduced) :
  n_days =90
  current_date = datetime.now()
  DATE_UPDATE_optuna = current_date.strftime('%d%m%Y')
  PIPELINE, train_transformed, covariates_transformed, create_covariates, train_darts, test_darts, total_darts = PrepareDataForDarts(df_final_test_singleProduct, df_final_train_singleProduct, df_final_reduced_singleProduct)
  tft_produit_result = function_result_final(nb_epoch, num_id, df_final_test_singleProduct, df_final_train_singleProduct, df_final_reduced_singleProduct)
  study_name_dateAujoudhui = "ErrorProduit_detail" + f"{num_id}" +f"_{DATE_UPDATE_optuna}"
  save_csv_to_gcs(study_name_dateAujoudhui, tft_produit_result)