
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

def handle_missing_data(df: pd.DataFrame, columns: list = None, strategy: str = "empty_string") -> pd.DataFrame:
    """Manage null dataframe values, different strategies. Apply only to specific columns if provided."""
    
    cols_to_clean = columns if columns else df.columns
    
    if strategy == "empty_string":
        df[cols_to_clean] = df[cols_to_clean].fillna("")
    elif strategy == "drop":
        df = df.dropna(subset=cols_to_clean)
    elif strategy == "placeholder":
        df[cols_to_clean] = df[cols_to_clean].fillna("missing_data")
    
    return df

def concatenate_df(df: pd.DataFrame, columns: list):
    """Concatenate multiple text columns into a single space-separated column."""

    df = handle_missing_data(df, columns=columns, strategy="empty_string")
    
    new_col_name = "_".join(columns) 
    df[new_col_name] = df[columns].astype(str).agg(" ".join, axis=1)

    
    return df

def ingestion(
    data_url: str,
    expected_columns: list,
    text_columns: list = None,
    target_column: str = None,
    value_true: str = None,
    value_false: str = None
) -> pd.DataFrame:
    """
    Execute the complete data ingestion pipeline.

    Steps
    -----
    1. Validate schema.
    2. Load dataset.
    3. Handle missing values.
    4. Concatenate text columns.
    5. Map target labels.

    Returns
    -------
    pd.DataFrame
        Processed dataframe ready for preprocessing.
    """

    #if not schema_validation(data_url, expected_columns):
    #    raise ValueError("Dataset schema is not valid.")

    df = data_loader(data_url,expected_columns )

    df = handle_missing_data(df)

    if text_columns:
        df = concatenate_df(df, text_columns)

    if (
        target_column
        and value_true is not None
        and value_false is not None
    ):
        df = data_mapping(
            df,
            target_column,
            value_true,
            value_false
        )

    print("Data ingestion completed.")

    return df