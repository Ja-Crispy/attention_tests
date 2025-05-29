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
# Ensure logger is configured at application entry point or use a getLogger approach
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
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

    files_exist = all((raw_data_path / fname).exists() for fname in EXPECTED_RAW_FILES)
    if files_exist:
        logger.info(f"WikiText-103 raw files already exist in {raw_data_dir}. Skipping download.")
        return

    logger.info(f"Downloading WikiText-103 raw data from {WIKITEXT103_URL}...")
    try:
        response = requests.get(WIKITEXT103_URL, stream=True)
        response.raise_for_status() 

        logger.info("Download complete. Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for member_info in z.infolist():
                parts = Path(member_info.filename).parts
                if len(parts) > 1 and parts[0] == "wikitext-103-raw" and parts[1] in EXPECTED_RAW_FILES:
                    file_content = z.read(member_info.filename)
                    target_path = raw_data_path / parts[1]
                    with open(target_path, 'wb') as f:
                        f.write(file_content)
                    logger.info(f"Extracted {parts[1]} to {target_path}")
        
        logger.info(f"WikiText-103 raw files successfully downloaded and extracted to {raw_data_dir}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading WikiText-103: {e}")
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting zip file for WikiText-103: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during download/extraction: {e}")
        raise


def preprocess_text_files_for_causal_lm(
    raw_data_dir: str, 
    processed_data_dir: str, 
    tokenizer_name_or_path: str, 
    config: dict
) -> None:
    """
    Processes raw text files (WikiText-103 format) for Causal Language Modeling.
    Reads raw files line by line, tokenizes each line, and concatenates the results.
    The entire tokenized content for each split is saved as a single JSON line 
    in a corresponding .jsonl file.

    Args:
        raw_data_dir (str): Directory containing raw text files (wiki.train.raw, etc.).
        processed_data_dir (str): Directory to save the processed .jsonl files.
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer.
        config (dict): Configuration dictionary, expected to contain:
                       `config['data']['tokenizer_max_length']` (used as context for tokenizer loading,
                       but individual lines are not truncated by this value during line-by-line tokenization).
    """
    logger.info(f"Preprocessing text files from {raw_data_dir} to {processed_data_dir} for Causal LM (memory-efficient)...")
    Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_max_len_config = config.get('data', {}).get('tokenizer_max_length', 1024) 
    logger.info(f"Using tokenizer: {tokenizer_name_or_path}. Configured tokenizer_max_length for context: {tokenizer_max_len_config}")
    
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len_config)

    split_to_raw_filename = {
        "train": "wiki.train.raw",
        "valid": "wiki.valid.raw",
        "test": "wiki.test.raw"
    }

    for split_name, raw_filename in split_to_raw_filename.items():
        raw_file_path = Path(raw_data_dir) / raw_filename
        processed_file_path = Path(processed_data_dir) / f"{split_name}.jsonl"

        if not raw_file_path.exists():
            logger.warning(f"Raw file {raw_file_path} not found. Skipping preprocessing for this split.")
            continue

        logger.info(f"Processing {raw_file_path} line by line...")
        
        concatenated_input_ids = []
        concatenated_attention_masks = []
        num_lines_processed = 0

        with open(raw_file_path, 'r', encoding='utf-8') as raw_file:
            for line in raw_file:
                processed_line = line.strip()
                if not processed_line:
                    continue 
                
                num_lines_processed += 1
                if num_lines_processed % 100000 == 0: 
                    logger.info(f"  Processed {num_lines_processed} lines from {raw_file_path}...")

                tokenized_segment = tokenize_function(
                    processed_line, 
                    tokenizer, 
                    max_length=tokenizer_max_len_config, 
                    padding_strategy="do_not_pad", 
                    truncation_strategy=False 
                )
                
                concatenated_input_ids.extend(tokenized_segment["input_ids"])
                concatenated_attention_masks.extend(tokenized_segment["attention_mask"])

        logger.info(f"Finished tokenizing {raw_filename} for {split_name} split. Total lines processed: {num_lines_processed}.")
        logger.info(f"For {split_name} split: Generated {len(concatenated_input_ids)} total tokens.") # Log Token Counts per Split
        
        if split_name == "valid":
            sample_size = 50 
            logger.info(f"Sample of 'valid' split input_ids (first {sample_size} tokens): {concatenated_input_ids[:sample_size]}")
            logger.info(f"Sample of 'valid' split attention_mask (first {sample_size} tokens): {concatenated_attention_masks[:sample_size]}")

        if not concatenated_input_ids:
            logger.warning(f"No tokens were generated for {raw_file_path}. Output file will be empty or contain an empty list.")
        
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json_record = {"input_ids": concatenated_input_ids, "attention_mask": concatenated_attention_masks}
            f.write(json.dumps(json_record) + '\n')
        
        logger.info(f"Saved processed {split_name} data to {processed_file_path}")

    logger.info("Preprocessing complete.")


class CausalLMTrainingDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int, pad_token_id: int):
        logger.info(f"Initializing CausalLMTrainingDataset from: {file_path}, sequence_length: {sequence_length}")
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
        if self.input_ids.size(0) == 0 : 
            logger.warning(f"The file {file_path} resulted in zero tokens after processing. Dataset will be empty or padded.")
        
        if self.input_ids.size(0) < min_tokens and self.input_ids.size(0) > 0 : 
            pad_len = min_tokens - self.input_ids.size(0)
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros((pad_len,), dtype=torch.long) 
            self.input_ids = torch.cat([self.input_ids, pad_ids], dim=0)
            self.attention_mask = torch.cat([self.attention_mask, pad_attn], dim=0)
            logger.info(f"Padded dataset to {min_tokens} tokens for sequence length {self.sequence_length}.")
        
        if self.input_ids.size(0) == 0: 
            self.num_examples = 0
        else:
            self.num_examples = (self.input_ids.size(0) - 1) // self.sequence_length
        
        logger.info(f"Dataset loaded. Total tokens: {self.input_ids.size(0)}. Num examples for seq_len {self.sequence_length}: {self.num_examples}")


    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples: 
            raise IndexError("Index out of bounds")

        start_index = idx * self.sequence_length
        end_index = start_index + self.sequence_length + 1
        
        actual_end_index = min(end_index, self.input_ids.size(0))
        full_chunk = self.input_ids[start_index:actual_end_index]
        
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
    if not logging.getLogger(__name__).hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Text Dataset Utilities example usage (WikiText-103 with enhanced logging):") # Updated print to logger.info
    
    main_example_dir_name = "wikitext103_dataset_example_run_mem_efficient_logged" 
    base_dir = Path("./data") / main_example_dir_name
    raw_dir = base_dir / "raw_wikitext103" 
    processed_dir = base_dir / "processed_wikitext103_for_causal_lm" 
    
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    example_config = {
        'data': {
            'sequence_length': 64, 
            'tokenizer_max_length': 1024 
        }
    }
    tokenizer_name_for_example = "gpt2" 

    try:
        download_raw_text_files(str(raw_dir))
        if not all((Path(raw_dir) / fname).exists() for fname in EXPECTED_RAW_FILES):
            logger.error(f"Error: Not all expected files were found in {raw_dir} after download attempt.") # Changed print to logger.error
        else:
            logger.info(f"Successfully found expected raw files in {raw_dir}") # Changed print to logger.info
            preprocess_text_files_for_causal_lm(
                str(raw_dir), str(processed_dir), tokenizer_name_for_example, example_config)

            # Add a Final Verification Message (as per prompt)
            logger.info("Successfully finished preprocessing all splits with line-by-line processing.")
            logger.info("Please manually check the output .jsonl files in the processed data directory "
                        f"({processed_dir}) to ensure correct format (single JSON line, "
                        "'input_ids' and 'attention_mask' as flat lists).")


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
                    logger.info(f"\nCreated CausalLMTrainingDataset with {len(dataset)} examples.") # Changed print to logger.info
                    first_batch = next(iter(dataloader))
                    logger.info("\nFirst batch details:") # Changed print to logger.info
                    logger.info(f"  Input IDs shape: {first_batch['input_ids'].shape}")
                    logger.info(f"  Attention Mask shape: {first_batch['attention_mask'].shape}")
                    logger.info(f"  Labels shape: {first_batch['labels'].shape}")
                else:
                    logger.info("Dataset created but contains no examples.") # Changed print to logger.info
            else:
                logger.info(f"Processed training file {train_jsonl_path} not found.") # Changed print to logger.info
    except Exception as e:
        logger.error(f"An error occurred in the example usage: {e}", exc_info=True) # Changed print to logger.error
        # import traceback # Already imported via exc_info=True
        # traceback.print_exc() # exc_info=True handles this
    finally:
        if base_dir.exists():
            logger.info(f"\nCleanup of {base_dir} can be done manually for inspection.") # Changed print to logger.info
        else:
            logger.info(f"\nBase directory {base_dir} does not exist, no cleanup needed or already cleaned.") # Changed print to logger.info
    logger.info("\nText Dataset Utilities example finished.") # Changed print to logger.info
