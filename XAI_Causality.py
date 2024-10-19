# Dans cette feuille de calcule nous avons les outils pour trouvé les relations entre les produits :
# Il y a les méthodes de découverte causales et XAI

import shap
import tigramite
from numpy import np
from pandas import pd
from sklearn.preprocessing import LabelEncoder
from tigramite import data_processing as pp
from tigramite.independence_tests import parcorr
from tigramite.pcmci import PCMCI


def create_feature_importance_table(
    df_final_train_shap, shap_values_df, list_produit_test, feature_comparition
):
    all_feature_importance = []

    for unique_id in [
        x for x in df_final_train_shap["unique_id"].unique() if x in list_produit_test
    ]:
        shap_values_for_id = shap_values_df.loc[unique_id]

        # Calculate the mean absolute SHAP value for each feature
        feature_importance = np.abs(shap_values_for_id).mean(axis=0)
        total_importance = feature_importance.sum()
        feature_importance /= total_importance

        # Create a dictionary with feature importances
        importance_dict = {
            "unique_id": f"{unique_id}_xgb",
            **dict(zip(feature_comparition, feature_importance)),
        }

        all_feature_importance.append(importance_dict)

    # Create the final DataFrame
    final_feature_importance_df = pd.DataFrame(all_feature_importance)

    # Set 'unique_id' as the index
    final_feature_importance_df.set_index("unique_id", inplace=True)

    return final_feature_importance_df


###########################################################
#### Causal Discovery ####
###########################################################


def prepare_data_for_tigramite(df):
    # 1. Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # 2. Handle non-numeric data
    for col in non_numeric_columns:
        if df[col].dtype == "object":
            # Use LabelEncoder for categorical data
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            # Convert other types to numeric if possible
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Handle any remaining NaN values
    df = df.fillna(df.mean())

    # 4. Ensure all data is float
    df = df.astype(float)

    # 5. Create tigramite DataFrame
    pp_frame = pp.DataFrame(data=df.values, var_names=df.columns.tolist())

    return pp_frame


def time_averaged_tigramite_results(results, var_names, threshold=0.1):
    val_matrix = results["val_matrix"]

    # Sum across all lags and normalize by the number of lags
    val_matrix_2d = np.sum(val_matrix, axis=2) / val_matrix.shape[2]

    # Create a DataFrame for easier interpretation
    mi_df = pd.DataFrame(val_matrix_2d, index=var_names, columns=var_names)

    # Get the upper triangle of the matrix (excluding diagonal)
    upper_tri = np.triu(mi_df.values, k=1)

    # Create a list of tuples: (var1, var2, MI value)
    mi_pairs = [
        (var_names[i], var_names[j], upper_tri[i, j])
        for i in range(len(var_names))
        for j in range(i + 1, len(var_names))
        if upper_tri[i, j] > threshold
    ]

    # Sort by MI value in descending order
    mi_pairs.sort(key=lambda x: x[2], reverse=True)

    return mi_pairs, mi_df


def interpret_3d_tigramite_results(results, var_names, threshold=0.1, lag=0):
    val_matrix = results["val_matrix"]

    # Extract the 2D slice for the specified lag
    val_matrix_2d = val_matrix[:, :, lag]

    # Create a DataFrame for easier interpretation
    mi_df = pd.DataFrame(val_matrix_2d, index=var_names, columns=var_names)

    # Get the upper triangle of the matrix (excluding diagonal)
    upper_tri = np.triu(mi_df.values, k=1)

    # Create a list of tuples: (var1, var2, MI value)
    mi_pairs = [
        (var_names[i], var_names[j], upper_tri[i, j])
        for i in range(len(var_names))
        for j in range(i + 1, len(var_names))
        if upper_tri[i, j] > threshold
    ]

    # Sort by MI value in descending order
    mi_pairs.sort(key=lambda x: x[2], reverse=True)

    return mi_pairs, mi_df


def calcule_rolling_mean_lag(df_for_output, y):
    rolling_mean = 0
    for x in df_for_output.index:
        if type(x) == int and x < y:
            rolling_mean += df_for_output["importance"][x]
    rolling_mean_lag = rolling_mean / y
    return rolling_mean_lag


def calcule_rolling_max_lag(df_for_output, y):
    rolling_max = 0
    for x in df_for_output.index:
        if type(x) == int and x < y:
            if rolling_max < df_for_output["importance"][x]:
                rolling_max = df_for_output["importance"][x]
    return rolling_max


def calcule_rolling_min_lag(df_for_output, y):
    rolling_min = df_for_output["importance"][y]
    for x in df_for_output.index:
        if type(x) == int and x < y:
            if rolling_min > df_for_output["importance"][x]:
                rolling_min = df_for_output["importance"][x]
    return rolling_min


def prepar_for_output(df_for_output):
    new_index_entries = {}

    for x in df_for_output.index:
        if isinstance(x, int):
            if x in [1, 7, 15, 30]:
                new_index_entries[f"lag{x}"] = df_for_output["importance"][x]

            if x == 7:
                new_index_entries[f"rolling_mean_lag1_window_size{x}"] = (
                    calcule_rolling_mean_lag(x)
                )
                new_index_entries[f"rolling_max_lag1_window_size{x}"] = (
                    calcule_rolling_max_lag(x)
                )
                new_index_entries[f"rolling_min_lag1_window_size{x}"] = (
                    calcule_rolling_min_lag(x)
                )

            if x == 15:
                new_index_entries[f"rolling_mean_lag2_window_size{x}"] = (
                    calcule_rolling_mean_lag(x)
                )
                new_index_entries[f"rolling_max_lag2_window_size{x}"] = (
                    calcule_rolling_max_lag(x)
                )
                new_index_entries[f"rolling_min_lag2_window_size{x}"] = (
                    calcule_rolling_min_lag(x)
                )

            if x == 30:
                new_index_entries[f"rolling_mean_lag3_window_size{x}"] = (
                    calcule_rolling_mean_lag(x)
                )
                new_index_entries[f"rolling_max_lag3_window_size{x}"] = (
                    calcule_rolling_max_lag(x)
                )
                new_index_entries[f"rolling_min_lag3_window_size{x}"] = (
                    calcule_rolling_min_lag(x)
                )

    # Create a new DataFrame with the new index entries
    new_rows = pd.DataFrame(new_index_entries, index=["importance"]).T

    # Concatenate the new rows to the existing DataFrame
    df_for_output = pd.concat([df_for_output, new_rows])
    return df_for_output


def analyze_quantity_effects(
    results, var_names, alpha_level=0.05, min_effect_size=0.1, max_lag=5
):
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]
    quantity_index = var_names.index("QUANTITE")

    quantity_effects = []

    for lag in range(max_lag):
        for i, var in enumerate(var_names):
            if var != "QUANTITE":
                effect_value = val_matrix[i, quantity_index, lag]
                p_value = p_matrix[i, quantity_index, lag]

                if p_value <= alpha_level and abs(effect_value) >= min_effect_size:
                    quantity_effects.append(
                        {
                            "cause": var,
                            "lag": lag,
                            "effect_value": effect_value,
                            "p_value": p_value,
                        }
                    )

    # Convert to DataFrame and sort
    effects_df = pd.DataFrame(quantity_effects)
    effects_df = effects_df.sort_values("effect_value", key=abs, ascending=False)

    return effects_df


def causal_impact_on_quantity(causal_df, var_names, max_lag=5):
    quantity_impacts = []

    for lag in range(max_lag):
        # Filter the DataFrame for the specific lag
        lag_df = causal_df[causal_df["lag"] == lag]

        # Extract impacts on QUANTITE
        quantity_effects = lag_df[lag_df["effect"] == "QUANTITE"]

        for _, link in quantity_effects.iterrows():
            quantity_impacts.append(
                {
                    "cause": link["cause"],
                    "lag": lag,
                    "value": link["value"],
                    "p_value": link["p_value"],
                }
            )

    # Convert to DataFrame for easier manipulation
    impact_df = pd.DataFrame(quantity_impacts)

    # Sort by absolute value of impact
    impact_df["abs_value"] = impact_df["value"].abs()
    impact_df = impact_df.sort_values("abs_value", ascending=False).drop(
        "abs_value", axis=1
    )

    return impact_df


def sum_absolute_by_variable(data, variable=None):
    # Initialize a dictionary to store sums for each variable
    sums = {}

    # Iterate through each tuple in the data
    for _, series in data:
        # For each variable in the series
        for var, value in series.items():
            # Add the absolute value to the sum for this variable
            sums[var] = sums.get(var, 0) + abs(value)

    # If a specific variable is requested, return only that sum
    if variable:
        return sums.get(variable, 0)

    # Otherwise, return all sums
    return sums


def UnselectedColunnes(df):
    unselect = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            unselect.append(column)
    return unselect


def analyse_with_MIC(df_final_reduced, unique_id):
    tau_max = 30
    pc_alpha = 0.01
    unselect = UnselectedColunnes(
        df_final_reduced[df_final_reduced["unique_id"] == unique_id]
    )
    quantite_uniqueProduit = df_final_reduced[
        df_final_reduced["unique_id"] == unique_id
    ][
        [
            x
            for x in df_final_reduced.columns
            if x not in unselect + ["PARAM_MARGE", "TVA", "ds", "unique_id"]
        ]
    ]

    parcorr_test = parcorr.ParCorr()
    pp_frame = prepare_data_for_tigramite(quantite_uniqueProduit)

    pcmci = PCMCI(dataframe=pp_frame, cond_ind_test=parcorr_test, verbosity=1)
    pcmci.verbosity = 2

    print("\nPCMCI object created successfully!")
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
    Laged_frame = []
    importance_lag = []

    for lag in range(results["val_matrix"].shape[2]):
        mi_pairs, mi_df = interpret_3d_tigramite_results(
            results,
            [x for x in quantite_uniqueProduit.columns if x not in ["unique_id", "ds"]],
            lag=lag,
        )
        # Add lag number as index
        Laged_frame.append((lag, mi_df["QUANTITE"]))
        importance_lag.append((lag, sum(abs(num) for num in mi_df["QUANTITE"])))
    sum_test = sum_absolute_by_variable(Laged_frame)
    total_sum_test = sum(sum_test.values())
    importance_MIC = [(var, sum_test[var] / total_sum_test) for var in sum_test]
    total_importance = sum(abs(x[1]) for x in importance_lag)
    importance_lag_total = [(x[0], x[1] / total_importance) for x in importance_lag]

    df_for_output = pd.DataFrame(
        importance_MIC + importance_lag_total, columns=["variable", "importance"]
    )
    df_for_output = df_for_output.set_index("variable")
    df_id = prepar_for_output(df_for_output)
    importance_dict = {
        "unique_id": f"{unique_id}_PCMIC",
        **dict(zip(df_id.index, df_id["importance"])),
    }
    return importance_dict
