
import pandas as pd

def schema_validation(data_url: str, columns: list) -> bool:
    """Validate the initial schema structure for the data"""
    try: 
        df = pd.read_csv(data_url, nrows=0) # solo leemos la primera fila
    except:
        print("dataframe unable to read csv")
        return False

    dataframe_columns = df.columns.to_list()    

    return dataframe_columns == columns
def data_loader(data_url: str, columns: list) -> pd.DataFrame: 
    """Load the file data in a pandas dataframe"""
    #is valid?
    if not schema_validation(data_url,  columns):
        print("Data schema not valid")
        return None
    print("Data Loaded, schema valid")
    return pd.read_csv(data_url) 

def data_mapping(df: pd.DataFrame, column: str, value_true:str, value_false:str):
    df[column] = df[column].map({value_true: 1, value_false: 0})
    return df

def concatenate_df(df: pd.DataFrame, columns: list):
    new_col_name = "".join(map(str, columns))
    
    # concatenamos
    df[new_col_name] = df[columns].astype(str).agg("".join, axis=1)
    
    return df

#manejar datos vacios en caso de ser necesario


