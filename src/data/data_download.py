import os

from sklearn import datasets
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

if __name__ == "__main__":  
    dataset = datasets.load_diabetes()
    features = pd.DataFrame(data=dataset.data, 
                            columns=["feat%s" % x for x in range(dataset.data.shape[1])])
    target = pd.DataFrame(data=dataset.target, columns=["target"])

    features.to_csv("%s/initial_data.csv" % os.environ.get("INITIAL_DATA_PATH"))
    target.to_csv("%s/target.csv" % os.environ.get("INITIAL_DATA_PATH"))
