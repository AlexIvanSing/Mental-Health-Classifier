
import pandas as pd
from sklearn.model_selection import train_test_split


def schema_validation(data_url: str, columns: list) -> bool:
    """
    Verify that the CSV header matches the expected column list.

    Reads only the header row to avoid loading the full file. Used as a
    fast gate before `data_loader` to catch upstream schema drift early.

    Parameters
    ----------
    data_url : str
        Path or URL to the CSV file.
    columns : list of str
        Expected column names, in order.

    Returns
    -------
    bool
        True if the schema matches; False otherwise (or on read error).
    """
    try:
        df = pd.read_csv(data_url, nrows=0)
    except Exception:
        print("dataframe unable to read csv")
        return False

    dataframe_columns = df.columns.to_list()

    return dataframe_columns == columns


def data_loader(data_url: str, columns: list) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame after validating its schema.

    Parameters
    ----------
    data_url : str
        Path or URL to the CSV file.
    columns : list of str
        Expected column names, in order.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None if schema validation fails.
    """
    if not schema_validation(data_url, columns):
        print("Data schema not valid")
        return None
    print("Data Loaded, schema valid")
    return pd.read_csv(data_url)


def data_mapping(df: pd.DataFrame, column: str, value_true: str, value_false: str) -> pd.DataFrame:
    """
    Map a binary categorical column into 0/1 integers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to map.
    value_true : str
        Source value that maps to 1.
    value_false : str
        Source value that maps to 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with `column` converted to integer labels.
    """
    df[column] = df[column].map({value_true: 1, value_false: 0})
    return df


def handle_missing_data(df: pd.DataFrame, columns: list = None, strategy: str = "empty_string") -> pd.DataFrame:
    """
    Resolve null values in a DataFrame using one of several strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Subset of columns to clean. If None, applies to all columns.
    strategy : {"empty_string", "drop", "placeholder"}
        - "empty_string": fill NaNs with "".
        - "drop": drop rows where any selected column is NaN.
        - "placeholder": fill NaNs with the literal "missing_data".

    Returns
    -------
    pd.DataFrame
        DataFrame with nulls resolved.
    """
    
    cols_to_clean = columns if columns else df.columns
    
    if strategy == "empty_string":
        df[cols_to_clean] = df[cols_to_clean].fillna("")
    elif strategy == "drop":
        df = df.dropna(subset=cols_to_clean)
    elif strategy == "placeholder":
        df[cols_to_clean] = df[cols_to_clean].fillna("missing_data")
    
    return df

def concatenate_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Concatenate multiple text columns into one space-separated column.

    The new column is named by joining the source column names with `_`
    (e.g., `title` + `text` → `title_text`). Nulls in the source columns
    are coerced to empty strings before concatenation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to concatenate, in order.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new concatenated column appended.
    """

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

    df = data_loader(data_url, expected_columns)
    if df is None:
        return None

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


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a raw DataFrame into stratified train/test CSVs on disk.

    Intended to be run **once** before any training or inference, so that
    every downstream run reads from the same frozen splits. Stratifies on
    `target_column` to preserve class balance across the split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_column : str
        Column used for stratified splitting.
    test_size : float, default 0.2
        Fraction of the dataset reserved for testing.
    random_state : int, default 42
        Seed for reproducibility.
    train_path : str, default "data/train.csv"
        Output path for the training split.
    test_path : str, default "data/test.csv"
        Output path for the test split.

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, test_df) — also written to disk.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Test:  {len(test_df)} rows -> {test_path}")

    return train_df, test_df