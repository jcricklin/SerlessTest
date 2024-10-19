# Optimisation des modèles : 
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import(
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import xgboost 
from window_ops.rolling import rolling_mean, rolling_max, rolling_min, rolling_cv
import pandas as pd
import numpy as np
from mlforecast import MLForecast  
from mlforecast.lag_transforms import ExpandingMean, RollingMean, SeasonalRollingMean
import time
from datetime import datetime

from utils import add_ds_unique_id
from utils import split_df
from utils import control_timeId_Matching
from utils import root_mean_square_error
from utils import custom_train_val_split
from utils import sauvtoGCS

import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState


# from darts import TimeSeries
# from darts.dataprocessing.pipeline import Pipeline
# from darts.models import TFTModel
# from darts.dataprocessing.transformers import Scaler
# from darts.utils.timeseries_generation import datetime_attribute_timeseries
# from darts.utils.likelihood_models import QuantileRegression
# from darts.dataprocessing.transformers import StaticCovariatesTransformer, MissingValuesFiller


class LastPlacePruner(BasePruner):
    def __init__(self, warmup_steps, warmup_trials):
        self._warmup_steps = warmup_steps
        self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]

            # Get scores from other trials in the study reported at the same step
            completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            other_scores = sorted(other_scores)

            # Prune if this trial at this step has a lower value than all completed trials
            # at the same step. Note that steps will begin numbering at 0 in the objective
            # function definition below.
            if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
                if this_score < other_scores[0]:
                    print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                    return True

        return False


def objectiveTest0(trial, df_final):
    # Ensure exogenous columns do not include the target 'QUANTITE'
    exogenous = [x for x in df_final.columns if x not in ['QUANTITE', 'DATE_IMPORT', 'ID_PRODUIT', 'unique_id', 'ds']]

    df_reset = df_final.copy()

    df_reset = add_ds_unique_id(df_reset)

    early_stopping_rounds = 100

    param = {
        "tree_method" : 'hist',
        "predictor" : 'predictor',
        "objective": trial.suggest_categorical("objective", ["reg:tweedie", "reg:absoluteerror", "reg:squarederror"]),
        "eval_metric": trial.suggest_categorical("eval_metric", ["rmse", "mae"]),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 9),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-2, 8e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-2, 8e-1, log=True),
        'verbosity': 0
    }
    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 7)
        param["eta"] = trial.suggest_float("eta", 1e-2, 8e-1, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])

    models = [xgboost.XGBRegressor(**param)]

    model = MLForecast(
        models=models,
        freq='D',
        lags=[1, 7, 15],
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
            2: [(rolling_mean, 15), (rolling_max, 15), (rolling_min, 15)],
        },
        date_features=['dayofweek', 'month'],
        num_threads=1
    )


    sample_size = min(100, len(df_reset['unique_id']))
    sampled_ids = np.random.choice(df_reset['unique_id'], sample_size, replace=False)


    df_reset = df_reset[df_reset['unique_id'].isin(sampled_ids)].copy()


    X_reset_train, X_reset_test = split_df(df_reset,90)


    model.fit(X_reset_train, id_col='unique_id', time_col='ds', target_col='QUANTITE', static_features=[])

    if not X_reset_test.empty and 'ds' in X_reset_test.columns:

        control_timeId_Matching(X_reset_test)

        p = model.predict(len(X_reset_test)//len(list(set(X_reset_train['unique_id']))), X_df=X_reset_test)#_complete[['unique_id', 'ds'] + exogenous])
        p.loc[p['XGBRegressor'] < 0, 'XGBRegressor'] = 0

        X_reset_test_complete = X_reset_test.dropna(subset=['QUANTITE'])

        error = root_mean_square_error(X_reset_test['QUANTITE'], p['XGBRegressor'])

        return error
    



  
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
  


def create_and_use_optuna_XGBoost( final_df, DATE_UPDATE_optuna, nomencalture, model_forecast, target, exogenous, id_produit):
    
    study_name = f"Optuna_Study_{nomencalture}_{datetime.now().strftime('%Y-%m')}_{model_forecast}_GSCMlforecastest"
    start_time = time.time()

    pruner = LastPlacePruner(warmup_steps=1, warmup_trials=5)
    study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner)
    
    
    if "xgboost"  in model_forecast :
        study.optimize(lambda trial: objectiveTest0(trial, final_df), n_trials=50, callbacks=[print_callback], n_jobs=-1, gc_after_trial=True)

    # Télécharger le fichier sur Google Drive

    end_time = time.time()

    # Calcul et affichage de la durée d'exécution
    execution_time = end_time - start_time
    print(f"Le temps d'exécution est de {execution_time} secondes.")

    best_params = study.best_params
    best_params.setdefault('verbosity', 0)
    
    
    if  "xgboost"  in model_forecast :
        model_XgbOptimRMSE = xgboost.XGBRegressor(**best_params)

        model = MLForecast(
            models=[model_XgbOptimRMSE],
            freq='D',
            lags=[1, 7, 15, 30],
            lag_transforms={
                1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                2: [(rolling_mean, 15), (rolling_max, 15), (rolling_min, 15)], 
                3: [(rolling_mean, 30), (rolling_max, 30), (rolling_min, 30)]
                
            },
            num_threads=1
        )


    df_final = add_ds_unique_id(final_df)
    df_final_train, df_final_test = split_df(df_final,90)
    # Train on the entire dataset
    model.fit(df_final_train, id_col='unique_id', time_col='ds', target_col='QUANTITE', static_features=[])

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))
        
    sauvtoGCS(f"Xgboost{id_produit}Model_MLForecast_test",model)
    
    return model, best_params, study, df_final_train, df_final_test









  


