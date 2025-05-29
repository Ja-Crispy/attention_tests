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
    raw_data_path = Path(raw_data_dir)
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    files_exist = all((raw_data_path / fname).exists() for fname in EXPECTED_RAW_FILES)
    if files_exist:
        logging.info(f"WikiText-103 raw files already exist in {raw_data_dir}. Skipping download.")
        return

    logging.info(f"Downloading WikiText-103 raw data from {WIKITEXT103_URL}...")
    try:
        response = requests.get(WIKITEXT103_URL, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        logging.info("Download complete. Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The zip file contains a top-level directory "wikitext-103-raw/"
            # We need to extract files from this directory into raw_data_dir directly.
            for member_info in z.infolist():
                # Check if the member is one of the files we want and is inside the top-level folder
                # e.g., "wikitext-103-raw/wiki.train.raw"
                parts = Path(member_info.filename).parts
                if len(parts) > 1 and parts[0] == "wikitext-103-raw" and parts[1] in EXPECTED_RAW_FILES:
                    # Extract the file, stripping the top-level directory
                    file_content = z.read(member_info.filename)
                    target_path = raw_data_path / parts[1]
                    with open(target_path, 'wb') as f:
                        f.write(file_content)
                    logging.info(f"Extracted {parts[1]} to {target_path}")
        
        logging.info(f"WikiText-103 raw files successfully downloaded and extracted to {raw_data_dir}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading WikiText-103: {e}")
        raise
    except zipfile.BadZipFile as e:
        logging.error(f"Error extracting zip file for WikiText-103: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during download/extraction: {e}")
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

    # Determine actual filenames based on splits
    # WikiText-103 uses 'train', 'valid', 'test' for its splits.
    # The raw files are e.g. 'wiki.train.raw'.
    # The output processed files will be 'train.jsonl', 'valid.jsonl', 'test.jsonl'.
    split_to_raw_filename = {
        "train": "wiki.train.raw",
        "valid": "wiki.valid.raw",
        "test": "wiki.test.raw"
    }

    for split, raw_filename in split_to_raw_filename.items():
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
