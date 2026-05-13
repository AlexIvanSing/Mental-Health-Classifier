import pandas as pd
import pytest
import os
import tempfile
from src.data_ingestion import schema_validation, data_loader, data_mapping, concatenate_df

# Fixture para crear un CSV temporal que se borra después de las pruebas
@pytest.fixture
def temp_csv():
    data = {
        "id": [1, 2, 3],
        "status": ["activo", "inactivo", "activo"],
        "nombre": ["Ana", "Bob", "Cris"],
        "apellido": ["Perez", "Gomez", "Sanz"]
    }
    df = pd.DataFrame(data)
    
    # Creamos un archivo temporal
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        df.to_csv(path, index=False)
        yield path
    finally:
        os.close(fd)
        os.remove(path)

# --- PRUEBAS ---

def test_schema_validation_success(temp_csv):
    expected_cols = ["id", "status", "nombre", "apellido"]
    assert schema_validation(temp_csv, expected_cols) is True

def test_schema_validation_fail(temp_csv):
    wrong_cols = ["id", "status"] # Faltan columnas
    assert schema_validation(temp_csv, wrong_cols) is False

def test_data_loader_valid(temp_csv):
    cols = ["id", "status", "nombre", "apellido"]
    df = data_loader(temp_csv, cols)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

def test_data_mapping(temp_csv):
    cols = ["id", "status", "nombre", "apellido"]
    df = data_loader(temp_csv, cols)
    
    # Mapeamos 'activo' a 1 e 'inactivo' a 0
    df = data_mapping(df, "status", "activo", "inactivo")
    
    assert df["status"].iloc[0] == 1
    assert df["status"].iloc[1] == 0
    assert df["status"].dtype == int

def test_concatenate_df(temp_csv):
    cols_to_check = ["id", "status", "nombre", "apellido"]
    df = data_loader(temp_csv, cols_to_check)
    
    # Concatenamos nombre y apellido
    cols_to_join = ["nombre", "apellido"]
    df = concatenate_df(df, cols_to_join)
    
    assert "nombreapellido" in df.columns
    assert df["nombreapellido"].iloc[0] == "AnaPerez"
    assert df["nombreapellido"].iloc[2] == "CrisSanz"