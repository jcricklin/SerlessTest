# Dans cette feuille, on va stocker les programmes pour gérer les données :

# import OptiModels
import io
import pickle
import tempfile

import cloudpickle
import fsspec
import numpy as np
import pandas as pd
from google.cloud import storage
from mlforecast import MLForecast


def sauvtoGCS(destination_blob_name, mon_objet):

    # 1. Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # 2. Sérialiser l'objet avec pickle et l'écrire dans le fichier temporaire
        pickle.dump(mon_objet, temp_file)
        temp_file_name = temp_file.name

    # 3. Initialiser le client GCS
    client = storage.Client()

    # 4. Spécifier le bucket et le nom du fichier de destination
    bucket_name = "wine_product_data_frame"

    # 5. Obtenir une référence au bucket
    bucket = client.bucket(bucket_name)

    # 6. Créer un nouvel objet blob et uploader le fichier
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(temp_file_name)

    print(
        f"Fichier {temp_file_name} uploadé vers {destination_blob_name} dans le bucket {bucket_name}."
    )
    return temp_file_name, destination_blob_name, bucket_name


def list_bucket_contents(bucket_name):
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(bucket_name))
    print(f"Contents of bucket '{bucket_name}':")
    for blob in blobs:
        print(f"- {blob.name}")
    return blobs


def read_csv_from_gcs(bucket_name, file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_text()
    return content


def load_xgboost_model_from_gcsII(blob):
    # Télécharger le contenu du blob
    content = blob.download_as_bytes()

    # Utiliser BytesIO pour créer un fichier-like object
    byte_stream = io.BytesIO(content)

    # Désérialiser le modèle
    try:
        model = pickle.load(byte_stream)
        if isinstance(model, MLForecast):
            print(f"Modèle MLForecast chargé avec succès: {blob.name}")
            return model
        else:
            print(f"Le fichier {blob.name} n'est pas un modèle MLForecast valide.")
            return None
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {blob.name}: {str(e)}")
        return None


def load_registered_mlforecast(nom_blobs):
    lis_blobs = list_bucket_contents("wine_product_data_frame")
    cpt = 0
    for x in lis_blobs:
        if nom_blobs in x.name:
            cpt += 1
            model = load_xgboost_model_from_gcsII(x)
            if model is not None:
                print(model)
            if cpt > 10:
                break
    return model


def add_sell_price(df_final_reduced):
    df_final_reduced["PRIX_VENTE"] = np.where(
        df_final_reduced["QUANTITE"] == 0,
        0,
        np.round(
            (
                df_final_reduced["PARAM_MARGE"] * (1 + df_final_reduced["TVA"])
                + df_final_reduced["PRIX_ACHAT"]
            )
            / df_final_reduced["QUANTITE"],
            2,
        ),
    )
    df_final_reduced = reprocessing_restatement(df_final_reduced)
    return df_final_reduced


def reprocessing_restatement(data_produit):
    for column in data_produit.columns:
        if any(keyword in column for keyword in ["PRIX", "MARGE", "TVA"]):
            series = data_produit[column].replace(0, np.nan)
            series_filled = series.ffill().bfill()
            series_filled = series_filled.fillna(0)
            data_produit[column] = series_filled

    if "QUANTITE" in data_produit.columns:
        data_produit.loc[data_produit["QUANTITE"] < 0, "QUANTITE"] = 0
    else:
        data_produit.loc[data_produit["y"] < 0, "y"] = 0

    return data_produit


def AddPricesInDf(data_magasin):
    unique_ids = data_magasin["ID_PRODUIT"].unique()
    processed_dataframes = []

    for id_produit in unique_ids:
        subset_df = data_magasin[data_magasin["ID_PRODUIT"] == id_produit]
        processed_subset_df = reprocessing_restatement(subset_df)
        processed_dataframes.append(processed_subset_df)

    result_df = pd.concat(processed_dataframes, ignore_index=True)
    return result_df


def good_date(df_final):
    df_final["DATE_IMPORT"] = df_final["DATE_IMPORT"].apply(
        lambda x: (
            pd.to_datetime(x, format="%Y-%m-%d")
            if len(x) == 10
            else pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
        )
    )
    df_final["DATE_IMPORT"] = df_final["DATE_IMPORT"].dt.strftime("%Y/%m/%d")
    return df_final


def add_ds_unique_id(df_reset):
    if "ds" not in df_reset.columns:
        df_reset["ds"] = pd.to_datetime(df_reset["DATE_IMPORT"])
    if "unique_id" not in df_reset.columns:
        df_reset["unique_id"] = df_reset["ID_PRODUIT"]

    df_reset = df_reset[
        [x for x in df_reset.columns if x not in ["ID_PRODUIT", "DATE_IMPORT"]]
    ]
    return df_reset


def charge_from_google_Drive(file_path):
    existing_df = pd.read_csv(file_path)
    df_final = AddPricesInDf(existing_df)
    df_final = good_date(df_final)
    return df_final


def split_df(x_final, nb_days):
    # Convert the 'ds' column to datetime type
    x_final["ds"] = pd.to_datetime(x_final["ds"])

    # Calculate the cutoff date for the test set (last 90 days)
    cutoff_date = x_final["ds"].max() - pd.Timedelta(days=nb_days)

    # Split the dataframe into training and test sets
    x_train = x_final[x_final["ds"] < cutoff_date]
    x_test = x_final[x_final["ds"] >= cutoff_date]

    return x_train, x_test


def control_timeId_Matching(X_df):
    # Create a list of unique ids
    unique_ids = X_df["unique_id"].unique()

    # Create a list of times in the forecasting horizon
    forecast_horizon = X_df["ds"]

    # Create a multi-index with all possible combinations of id and time
    index = pd.MultiIndex.from_product(
        [unique_ids, forecast_horizon], names=["unique_id", "ds"]
    )

    # Create a reference dataframe with one row for each combination of id and time
    ref_df = pd.DataFrame(index=index)

    X_df = pd.merge(ref_df, X_df, on=["unique_id", "ds"], how="left")

    if X_df.isna().sum().sum() > 0:
        raise ValueError("Found missing inputs in X_df")
    else:
        print("Tout est bon !")


def root_mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def custom_train_val_split(X, y, val_ratio=0.2):
    """
    Split the data into training and validation sets.
    """
    n = len(X)
    val_size = int(n * val_ratio)
    indices = np.random.permutation(n)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return (
        X.iloc[train_indices],
        X.iloc[val_indices],
        y.iloc[train_indices],
        y.iloc[val_indices],
    )


def create_features(df, target_col, id_col="unique_id"):
    if id_col:
        df = df.sort_values([id_col, "ds"])
    else:
        df = df.sort_values("ds")

    # Créer les lags
    df["lag1"] = df.groupby(id_col)[target_col].shift(1)
    df["lag7"] = df.groupby(id_col)[target_col].shift(7)
    df["lag15"] = df.groupby(id_col)[target_col].shift(15)
    df["lag30"] = df.groupby(id_col)[target_col].shift(30)

    # Créer les rolling features
    df["rolling_mean_lag1_window_size7"] = df.groupby(id_col)["lag7"].transform(
        lambda x: x.rolling(window=7).mean()
    )
    df["rolling_max_lag1_window_size7"] = df.groupby(id_col)["lag7"].transform(
        lambda x: x.rolling(window=7).max()
    )
    df["rolling_min_lag1_window_size7"] = df.groupby(id_col)["lag7"].transform(
        lambda x: x.rolling(window=7).min()
    )

    df["rolling_mean_lag2_window_size15"] = df.groupby(id_col)["lag15"].transform(
        lambda x: x.rolling(window=15).mean()
    )
    df["rolling_max_lag2_window_size15"] = df.groupby(id_col)["lag15"].transform(
        lambda x: x.rolling(window=15).max()
    )
    df["rolling_min_lag2_window_size15"] = df.groupby(id_col)["lag15"].transform(
        lambda x: x.rolling(window=15).min()
    )

    df["rolling_mean_lag3_window_size30"] = df.groupby(id_col)["lag30"].transform(
        lambda x: x.rolling(window=30).mean()
    )
    df["rolling_max_lag3_window_size30"] = df.groupby(id_col)["lag30"].transform(
        lambda x: x.rolling(window=30).max()
    )
    df["rolling_min_lag3_window_size30"] = df.groupby(id_col)["lag30"].transform(
        lambda x: x.rolling(window=30).min()
    )

    df_without_nan = df.dropna()

    return df_without_nan
