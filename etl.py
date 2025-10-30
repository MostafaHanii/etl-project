import pandas as pd 
import numpy as np 


def extract(path):
    df=pd.read_parquet(path)
    return df
def transform(df):
    df=df.dropna(axis=0)
    df=df.drop_duplicates()
    return df

    return df
def load(df,path):
    df.to_csv(path,index=False)
df=extract("yellow_tripdata_2025-09.parquet")
print(df.count())
print(df.isna().sum())
df=transform(df)
load(df,"taxi_data.csv")