import pandas as pd 
import numpy as np 


def extract(path):
    df=pd.read_csv(path)
    return df
def transform(df):
    df=df.dropna(axis=0)
    df=df.drop_duplicates()
    return df

    return df
def load(df,path):
    df.to_csv(path,index=False)

if __name__ == "__main__":
    df=extract("data.csv")
    print(df.count())
    print(df.isna().sum())
    df=transform(df)
    load(df,"output.csv")
    