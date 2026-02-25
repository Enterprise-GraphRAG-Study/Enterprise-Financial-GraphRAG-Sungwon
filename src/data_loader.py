from datasets import load_dataset, DatasetDict


def load_financial_dataset(dataset_name: str) -> DatasetDict:
    """
    Loads a financial dataset from Hugging Face.

    Args:
        dataset_name (str): The name or path of the dataset on Hugging Face.

    Returns:
        DatasetDict: The loaded dataset object.
    """
    try:
        dataset = load_dataset(dataset_name)
        print(f"Successfully loaded dataset: {dataset_name}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
