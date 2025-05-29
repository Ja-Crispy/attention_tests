# text_dataset_utils.py (conceptually, actual filename: wikitext103.py)
# Handles downloading/creating raw text, preprocessing for Causal LM, and serving via PyTorch Dataset.

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import shutil # For robust directory removal in examples
import json # For saving/loading processed data list
import os
import logging
import requests
import zipfile
import io

# Import from the project's tokenizer.py
from .tokenizer import load_tokenizer, tokenize_function 

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WIKITEXT103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
EXPECTED_RAW_FILES = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]

def download_raw_text_files(raw_data_dir: str) -> None:
    """
    Downloads and extracts the WikiText-103 raw text files into the specified directory.

    Args:
        raw_data_dir (str): Path to the directory where raw text files will be extracted.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("datasets library not found. Please install with: pip install datasets")
        raise ImportError("datasets library is required for downloading WikiText-103")
    
    raw_data_path = Path(raw_data_dir)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    train_file = raw_data_path / "train.txt"
    valid_file = raw_data_path / "valid.txt"
    test_file = raw_data_path / "test.txt"
    
    # Check if files already exist
    if train_file.exists() and valid_file.exists() and test_file.exists():
        logging.info("WikiText-103 files already exist. Skipping download.")
        return
    
    logging.info("Downloading WikiText-103 from Hugging Face datasets...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        
        # Save train split
        logging.info("Saving train split...")
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(example['text'] + '\n')
        
        # Save validation split
        logging.info("Saving validation split...")
        with open(valid_file, 'w', encoding='utf-8') as f:
            for example in dataset['validation']:
                f.write(example['text'] + '\n')
        
        # Save test split
        logging.info("Saving test split...")
        with open(test_file, 'w', encoding='utf-8') as f:
            for example in dataset['test']:
                f.write(example['text'] + '\n')
        
        logging.info(f"WikiText-103 dataset downloaded successfully to {raw_data_dir}")
        
    except Exception as e:
        logging.error(f"Error downloading WikiText-103 from Hugging Face: {e}")
        raise


def preprocess_text_files_for_causal_lm(
    raw_data_dir: str, 
    processed_data_dir: str, 
    tokenizer_name_or_path: str, 
    config: dict
) -> None:
    """
    Processes raw text files (WikiText-103 format) for Causal Language Modeling.
    Each raw text file (wiki.train.raw, etc.) is tokenized
    and its entire content is saved as a single JSON line in a corresponding .jsonl file.

    Args:
        raw_data_dir (str): Directory containing raw text files (wiki.train.raw, etc.).
        processed_data_dir (str): Directory to save the processed .jsonl files.
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer.
        config (dict): Configuration dictionary, expected to contain:
                       `config['data']['tokenizer_max_length']`
    """
    logging.info(f"Preprocessing text files from {raw_data_dir} to {processed_data_dir} for Causal LM...")
    Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_max_len = config.get('data', {}).get('tokenizer_max_length', 1024) # Default from original file
    logging.info(f"Using tokenizer: {tokenizer_name_or_path} with max_length: {tokenizer_max_len}")
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len)

    # Define the mapping of splits to file names
    splits = {
        'train': 'train.txt',  # Changed from 'wiki.train.raw'
        'valid': 'valid.txt',  # Changed from 'wiki.valid.raw' 
        'test': 'test.txt'     # Changed from 'wiki.test.raw'
    }
    
    for split, raw_filename in splits.items():
        raw_file_path = Path(raw_data_dir) / raw_filename
        processed_file_path = Path(processed_data_dir) / f"{split}.jsonl" # Output remains train.jsonl etc.

        if not raw_file_path.exists():
            logging.warning(f"Raw file {raw_file_path} not found. Skipping preprocessing for this split.")
            continue

        logging.info(f"Processing {raw_file_path}...")
        text_content = raw_file_path.read_text(encoding='utf-8')

        if not text_content.strip():
            logging.warning(f"Raw file {raw_file_path} is empty or contains only whitespace. Skipping.")
            continue
        
        tokenized_data = tokenize_function(
            text_content, 
            tokenizer, 
            max_length=tokenizer_max_len, 
            padding_strategy="do_not_pad", 
            truncation_strategy=True 
        )
        
        input_ids_list = tokenized_data["input_ids"].tolist()
        attention_mask_list = tokenized_data["attention_mask"].tolist()

        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json_record = {"input_ids": input_ids_list, "attention_mask": attention_mask_list}
            f.write(json.dumps(json_record) + '\n')
        
        logging.info(f"Saved processed data for {split} to {processed_file_path} ({len(input_ids_list)} tokens).")

    logging.info("Preprocessing complete.")


class CausalLMTrainingDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int, pad_token_id: int):
        logging.info(f"Initializing CausalLMTrainingDataset from: {file_path}, sequence_length: {sequence_length}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline() 
            if not line:
                raise ValueError(f"Processed data file {file_path} is empty.")
            try:
                data_record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON from {file_path}. Ensure it's a valid JSON line.")

        self.input_ids = torch.tensor(data_record['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(data_record['attention_mask'], dtype=torch.long)
        
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id 

        if self.input_ids.dim() != 1 or self.attention_mask.dim() != 1:
            raise ValueError("Loaded 'input_ids' and 'attention_mask' must be 1D tensors (lists in JSON).")
        if self.input_ids.size(0) != self.attention_mask.size(0):
            raise ValueError("'input_ids' and 'attention_mask' must have the same length.")

        min_tokens = self.sequence_length + 1
        if self.input_ids.size(0) < min_tokens:
            pad_len = min_tokens - self.input_ids.size(0)
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros((pad_len,), dtype=torch.long)
            self.input_ids = torch.cat([self.input_ids, pad_ids], dim=0)
            self.attention_mask = torch.cat([self.attention_mask, pad_attn], dim=0)
            logging.info(f"Padded dataset to {min_tokens} tokens for sequence length {self.sequence_length}.")

        self.num_examples = (self.input_ids.size(0) - 1) // self.sequence_length
        logging.info(f"Dataset loaded. Total tokens: {self.input_ids.size(0)}. Num examples for seq_len {self.sequence_length}: {self.num_examples}")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        start_index = idx * self.sequence_length
        end_index = start_index + self.sequence_length + 1
        full_chunk = self.input_ids[start_index:end_index]
        
        input_ids_chunk = full_chunk[:-1]
        target_ids_chunk = full_chunk[1:].clone() 
        attention_mask_chunk = self.attention_mask[start_index : start_index + self.sequence_length]
        target_ids_chunk.masked_fill_(attention_mask_chunk == 0, -100)
        
        return {
            "input_ids": input_ids_chunk, 
            "attention_mask": attention_mask_chunk, 
            "labels": target_ids_chunk
        }

if __name__ == '__main__':
    print("Text Dataset Utilities example usage (WikiText-103):")
    
    main_example_dir_name = "wikitext103_dataset_example_run" # Changed for clarity
    base_dir = Path("./data") / main_example_dir_name
    raw_dir = base_dir / "raw_wikitext103" # Changed for clarity
    processed_dir = base_dir / "processed_wikitext103_for_causal_lm" # Changed for clarity
    
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    example_config = {
        'data': {
            'sequence_length': 64, 
            'tokenizer_max_length': 1024 # Larger for real data, but example uses subset
        }
    }
    tokenizer_name_for_example = "gpt2" 

    try:
        download_raw_text_files(str(raw_dir))

        # Validate download
        if not all((Path(raw_dir) / fname).exists() for fname in EXPECTED_RAW_FILES):
            print(f"Error: Not all expected files were found in {raw_dir} after download attempt.")
        else:
            print(f"Successfully found expected raw files in {raw_dir}")

            preprocess_text_files_for_causal_lm(
                raw_data_dir=str(raw_dir),
                processed_data_dir=str(processed_dir),
                tokenizer_name_or_path=tokenizer_name_for_example,
                config=example_config
            )

            temp_tokenizer = load_tokenizer(tokenizer_name_for_example)
            pad_token_id_for_dataset = temp_tokenizer.pad_token_id
            if pad_token_id_for_dataset is None:
                pad_token_id_for_dataset = temp_tokenizer.eos_token_id if temp_tokenizer.eos_token_id is not None else 0
            del temp_tokenizer

            train_jsonl_path = processed_dir / "train.jsonl"
            if train_jsonl_path.exists():
                dataset = CausalLMTrainingDataset(
                    file_path=str(train_jsonl_path), 
                    sequence_length=example_config['data']['sequence_length'],
                    pad_token_id=pad_token_id_for_dataset
                )
                
                if len(dataset) > 0:
                    dataloader = DataLoader(dataset, batch_size=2)
                    print(f"\nCreated CausalLMTrainingDataset with {len(dataset)} examples.")
                    first_batch = next(iter(dataloader))
                    print("\nFirst batch details:")
                    print(f"  Input IDs shape: {first_batch['input_ids'].shape}")
                    print(f"  Attention Mask shape: {first_batch['attention_mask'].shape}")
                    print(f"  Labels shape: {first_batch['labels'].shape}")
                else:
                    print("Dataset created but contains no examples.")
            else:
                print(f"Processed training file {train_jsonl_path} not found.")

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if base_dir.exists():
            # shutil.rmtree(base_dir) # Comment out to inspect files after run
            print(f"\nClean up of {base_dir} skipped for inspection.")
        else:
            print(f"\nBase directory {base_dir} does not exist, no cleanup needed or already cleaned.")


    print("\nText Dataset Utilities example finished.")
