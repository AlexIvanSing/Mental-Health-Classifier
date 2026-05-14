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