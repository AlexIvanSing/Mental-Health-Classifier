# ============================================================================
# Utilidades generales
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import yaml


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file from disk.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)