import pandas as pd
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List

def json_to_dataframe(json_file_path: str) -> Optional[pd.DataFrame]:

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    except FileNotFoundError:
        print(f"Error: File {json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid json.")
        return None
    except Exception as e:
        print(f"unexpected error: {e}")
        return None
    
def split_val_test(df: pd.DataFrame,
                   *,
                   seed: int,
                   test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    
    return train_df, val_df, test_df

def get_necessary_columns(df: pd.DataFrame,
                          columns_to_keep: List[str] = ['text', 'language', 'target', 'source']) -> pd.DataFrame:
    return df[columns_to_keep]

def convert_to_huggingface_dataset(df: pd.DataFrame) -> Dataset:
    dataset =  Dataset.from_pandas(df)
    dataset = dataset.rename_column('target', 'label')
    return dataset

def get_dataset_dict(train_dataset: Dataset,
                     validation_dataset: Dataset,
                     test_dataset: Dataset,
                     no_dataleak_test_dataset: Dataset) -> DatasetDict:
    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
        'no_dataleak_test': no_dataleak_test_dataset
    })
    return final_dataset