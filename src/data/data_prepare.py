import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def fillna(dataset: pd.DataFrame) -> pd.DataFrame:

    prepare_dataset = dataset.copy()
    for i, column in enumerate(dataset.columns):
        if i % 2 == 0:
            prepare_dataset[column] = prepare_dataset[column] - 1
        else:
            prepare_dataset[column] = prepare_dataset[column] - 2
    
    return prepare_dataset

if __name__ == "__main__":
    dataset = pd.read_csv("%s/initial_data.csv" % os.environ.get("INITIAL_DATA_PATH"))
    prepared_dataset = fillna(dataset=dataset)
    prepared_dataset.to_csv("%s/prepared_data.csv" % os.environ.get("PREPARED_DATA_PATH"))
