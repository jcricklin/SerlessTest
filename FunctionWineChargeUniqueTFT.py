# Dans cette feuille on mettera toutes les fonctions pour lire les données :
import pandas as pd


# Sépare le jeu de donné test du jeu de donnée train
def split_df(x_final, n_days):
    # Convert the 'ds' column to datetime type
    x_final["ds"] = pd.to_datetime(x_final["ds"])

    # Calculate the cutoff date for the test set (last 90 days)
    cutoff_date = x_final["ds"].max() - pd.Timedelta(days=n_days)

    # Split the dataframe into training and test sets
    x_train = x_final[x_final["ds"] < cutoff_date]
    x_test = x_final[x_final["ds"] >= cutoff_date]

    return x_train, x_test


# Cette fonction permet de charger l'ensemble du jeux de donné et de selectionné un produit :
def ChargeDataBase_(id_produit):
    file_path = "https://storage.googleapis.com/wine_product_data_frame/new_wine_df"
    df_final = pd.read_csv(file_path)

    return df_final


def select_product(id_produit, df_final_reduced, n_days):
    df_final_train, df_final_test = split_df(df_final_reduced, n_days)
    df_final_train_singleProduct = df_final_train[
        df_final_train["unique_id"] == id_produit
    ]
    df_final_test_singleProduct = df_final_test[
        df_final_test["unique_id"] == id_produit
    ]
    df_final_singleProduct = df_final_reduced[
        df_final_reduced["unique_id"] == id_produit
    ]
    return (
        df_final_train_singleProduct,
        df_final_test_singleProduct,
        df_final_singleProduct,
    )
