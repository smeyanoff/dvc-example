import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from yaml import load, Loader
from dvclive import Live

load_dotenv()

if __name__ == "__main__":

    with open("configs/train_config.yml", "r") as conf:
        train_config = load(conf, Loader=Loader)["train_config"]

    np.random.seed(train_config["seed"])

    dataset = pd.read_csv("%s/prepared_data.csv" % os.environ.get("PREPARED_DATA_PATH"))
    target = pd.read_csv("%s/target.csv" % os.environ.get("INITIAL_DATA_PATH"))

    train_index, validation_index = train_test_split(dataset.index, 
                                                     test_size=train_config["validation_size"])

    train_index, test_index = train_test_split(train_index, 
                                               test_size=train_config["test_size"])

    model = LinearRegression()
    model.fit(dataset.loc[train_index], target.loc[train_index])

    train_MSE = mean_squared_error(target.loc[test_index], 
                                   model.predict(dataset.loc[test_index]))

    test_MSE = mean_squared_error(target.loc[train_index], 
                                  model.predict(dataset.loc[train_index]))

    validation_MSE = mean_squared_error(target.loc[validation_index], 
                                        model.predict(dataset.loc[validation_index]))
    
    with open("%s/linear_model.pickle" % os.environ.get("MODELS_PATH"), "wb") as mod:
        mod.write(pickle.dumps(model))
    
    with Live(save_dvc_exp=True) as live:
        live.log_params(train_config)
        live.log_artifact("%s/linear_model.pickle" % os.environ.get("MODELS_PATH"))
        live.log_metric("train_MSE", train_MSE)
        live.log_metric("test_MSE", test_MSE)
        live.log_metric("validation_MSE", validation_MSE)
