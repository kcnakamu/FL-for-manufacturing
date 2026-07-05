def get_dataset_yaml(factory_dir: str) -> str:
    """Return the path to the Ultralytics data.yaml for a client/factory folder."""
    return f"{factory_dir}/data.yaml"