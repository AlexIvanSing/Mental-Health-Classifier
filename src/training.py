from src.data_ingestion import ingestion

# variables reemplazar en variables de entonro despues

df = ingestion(
    data_url="../data/data_test_fold1.csv",
    expected_columns=[
        "user_id",
        "text_id",
        "title",
        "text",
        "is_suicide"
    ],
    text_columns=["title", "text"],
    target_column="is_suicide",
    value_true="yes",
    value_false="no" 
)
