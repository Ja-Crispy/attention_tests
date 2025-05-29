# text_dataset_utils.py (conceptually, actual filename: wikitext103.py)
# Handles downloading/creating raw text, preprocessing for Causal LM, and serving via PyTorch Dataset.

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import shutil # For robust directory removal in examples
import json # For saving/loading processed data list
import os
import logging
from datasets import load_dataset # Import for Hugging Face datasets

# Import from the project's tokenizer.py
from .tokenizer import load_tokenizer, tokenize_function 

# Setup basic logging (favoring the more robust setup from pas-attention)
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Ensure logger is configured only once
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXPECTED_RAW_FILES = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"] # From pas-attention

def download_raw_text_files(raw_data_dir: str) -> None:
    """
    Downloads and extracts the WikiText-103 raw text files into the specified directory
    using the Hugging Face datasets library.

    Args:
        raw_data_dir (str): Path to the directory where raw text files will be extracted.
    """
    raw_data_path = Path(raw_data_dir)
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Check if all expected files (with .raw extension) already exist
    all_files_exist = all((raw_data_path / fname).exists() for fname in EXPECTED_RAW_FILES)
    if all_files_exist:
        logger.info(f"All raw text files already exist in {raw_data_dir}. Skipping download.")
        return

    logger.info("Downloading WikiText-103 raw data using the 'datasets' library...")
    try:
        # Load the dataset from Hugging Face
        # Using 'wikitext-103-raw-v1' which is the raw version
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

        # Define mapping from dataset split names to our expected filenames
        split_to_filename = {
            "train": "wiki.train.raw",
            "validation": "wiki.valid.raw",
            "test": "wiki.test.raw",
        }

        for split_name, filename in split_to_filename.items():
            output_file_path = raw_data_path / filename
            logger.info(f"Processing and saving {split_name} data to {output_file_path}...")
            with open(output_file_path, "w", encoding="utf-8") as f:
                # The dataset has a 'text' column. We join all text entries with newlines.
                # Filter out empty lines that might be present in the dataset.
                lines_to_write = [text_item for text_item in dataset[split_name]['text'] if text_item.strip()]
                f.write("\n".join(lines_to_write))
            logger.info(f"Successfully saved {output_file_path}")

        logger.info("Successfully downloaded and processed all WikiText-103 raw files.")

    except Exception as e:
        logger.error(f"Error downloading or processing WikiText-103 using 'datasets': {e}")
        # Optionally, clean up partially downloaded files if an error occurs
        for fname in EXPECTED_RAW_FILES:
            if (raw_data_path / fname).exists():
                (raw_data_path / fname).unlink()
        raise # Re-raise the exception to signal failure


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
        raw_data_dir (str): Directory containing raw text files (e.g., wiki.train.raw).
        processed_data_dir (str): Directory to save the processed .jsonl files (e.g., train.jsonl).
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer.
        config (dict): Configuration dictionary, expected to contain:
                       `config['data']['tokenizer_max_length']`.
    """
    logger.info(f"Preprocessing text files from {raw_data_dir} to {processed_data_dir} for Causal LM (memory-efficient)...")
    Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_max_len_config = config.get('data', {}).get('tokenizer_max_length', 1024) 
    logger.info(f"Using tokenizer: {tokenizer_name_or_path}. Configured tokenizer_max_length for context: {tokenizer_max_len_config}")
    
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len_config)

    # Uses the raw filenames as defined in EXPECTED_RAW_FILES
    split_to_raw_filename = {
        "train": EXPECTED_RAW_FILES[0], # "wiki.train.raw"
        "valid": EXPECTED_RAW_FILES[1], # "wiki.valid.raw"
        "test": EXPECTED_RAW_FILES[2]   # "wiki.test.raw"
    }

    for split_name, raw_filename in split_to_raw_filename.items():
        raw_file_path = Path(raw_data_dir) / raw_filename
        processed_file_path = Path(processed_data_dir) / f"{split_name}.jsonl" # Output is train.jsonl etc.

        if not raw_file_path.exists():
            logger.warning(f"Raw file {raw_file_path} not found. Skipping preprocessing for this split.")
            continue

        logger.info(f"Processing {raw_file_path} line by line...")
        
        concatenated_input_ids = []
        concatenated_attention_masks = [] # Though likely all 1s with do_not_pad and no truncation per line
        num_lines_processed = 0

        with open(raw_file_path, 'r', encoding='utf-8') as raw_file:
            for line_idx, line in enumerate(raw_file): # Use enumerate for robust line counting
                processed_line = line.strip()
                if not processed_line: # Skip empty lines
                    continue 
                
                num_lines_processed += 1
                if num_lines_processed % 200000 == 0: # Log progress less frequently for large files
                    logger.info(f"  Processed {num_lines_processed} lines from {raw_file_path}...")

                tokenized_segment = tokenize_function(
                    processed_line, 
                    tokenizer, 
                    max_length=tokenizer_max_len_config, # This primarily acts as context for the tokenizer object
                    padding_strategy="do_not_pad", 
                    truncation_strategy=False # Do not truncate individual lines; we want the full document tokenized
                )
                
                concatenated_input_ids.extend(tokenized_segment["input_ids"].tolist()) # Ensure tolist() if tensors
                concatenated_attention_masks.extend(tokenized_segment["attention_mask"].tolist())

        logger.info(f"Finished tokenizing {raw_filename} for {split_name} split. Total lines processed: {num_lines_processed}.")
        logger.info(f"For {split_name} split: Generated {len(concatenated_input_ids)} total tokens.")
        
        if split_name == "valid": # For debugging, log sample of 'valid' data
            sample_size = 50 
            logger.info(f"Sample of 'valid' split input_ids (first {sample_size} tokens): {concatenated_input_ids[:sample_size]}")
            logger.info(f"Sample of 'valid' split attention_mask (first {sample_size} tokens): {concatenated_attention_masks[:sample_size]}")

        if not concatenated_input_ids:
            logger.warning(f"No tokens were generated for {raw_file_path}. Output file {processed_file_path} will contain an empty list.")
        
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
        self.attention_mask = torch.tensor(data_record['attention_mask'], dtype=torch.long) # Should be mostly 1s
        
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id 

        if self.input_ids.dim() != 1 or self.attention_mask.dim() != 1:
            raise ValueError("Loaded 'input_ids' and 'attention_mask' must be 1D tensors (lists in JSON).")
        if self.input_ids.size(0) != self.attention_mask.size(0):
            raise ValueError("'input_ids' and 'attention_mask' must have the same length.")

        min_tokens = self.sequence_length + 1 # Need N+1 tokens to create one N-length input and N-length target
        
        if self.input_ids.size(0) == 0 : 
            logger.warning(f"The file {file_path} resulted in zero tokens after processing. Dataset will be empty.")
            self.num_examples = 0
        elif self.input_ids.size(0) < min_tokens : 
            logger.warning(
                f"Dataset from {file_path} has {self.input_ids.size(0)} tokens, "
                f"which is less than required min_tokens ({min_tokens}) for sequence_length {self.sequence_length}. "
                "Dataset will be empty after attempting to create examples."
            )
            # Don't pad here, let num_examples become 0 if not enough tokens for even one sequence.
            # Padding was done by Jules in CausalLMTrainingDataset for dummy data, 
            # but for real data, it's better to report if a split is too small.
            self.num_examples = 0
        else:
            self.num_examples = (self.input_ids.size(0) - 1) // self.sequence_length
        
        logger.info(f"Dataset loaded. Total tokens: {self.input_ids.size(0)}. Num examples for seq_len {self.sequence_length}: {self.num_examples}")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples: 
            raise IndexError("Index out of bounds for CausalLMTrainingDataset")

        start_index = idx * self.sequence_length
        # Slice to get sequence_length + 1 tokens to form input and target
        end_index = start_index + self.sequence_length + 1
        
        # actual_end_index = min(end_index, self.input_ids.size(0)) # Not needed due to num_examples calculation
        full_chunk = self.input_ids[start_index:end_index]
        
        input_ids_chunk = full_chunk[:-1]
        target_ids_chunk = full_chunk[1:].clone() 
        
        # The attention_mask from the .jsonl file should reflect real tokens.
        # For line-by-line processing with do_not_pad, these should be all 1s unless a line was empty.
        attention_mask_chunk = self.attention_mask[start_index : start_index + self.sequence_length]
        
        # Labels should be -100 where the *original source for that token position was padding*.
        # Since we are chunking from a long sequence of real tokens, the attention_mask_chunk
        # should effectively be all 1s here. If there *was* padding in the original stream
        # (e.g. if tokenize_function used padding="max_length" for segments), this would be important.
        # For now, assuming attention_mask_chunk correctly reflects valid tokens.
        target_ids_chunk.masked_fill_(attention_mask_chunk == 0, -100) 
        
        return {
            "input_ids": input_ids_chunk, 
            "attention_mask": attention_mask_chunk, 
            "labels": target_ids_chunk
        }

if __name__ == '__main__':
    # Ensure logger is configured for __main__ execution
    if not logging.getLogger(__name__).hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Text Dataset Utilities example usage (WikiText-103 with memory-efficient preprocessing and enhanced logging):")
    
    main_example_dir_name = "wikitext103_dataset_example_run_final" 
    base_dir = Path("./data") / main_example_dir_name
    raw_dir = base_dir / "raw_wikitext103" 
    processed_dir = base_dir / "processed_wikitext103_for_causal_lm" 
    
    if base_dir.exists(): # Start fresh for example
        shutil.rmtree(base_dir)
    
    example_config = {
        'data': {
            'sequence_length': 64, 
            'tokenizer_max_length': 1024 # For tokenizer loading context, not per-line truncation
        }
    }
    tokenizer_name_for_example = "gpt2" 

    try:
        download_raw_text_files(str(raw_dir))
        
        if not all((Path(raw_dir) / fname).exists() for fname in EXPECTED_RAW_FILES):
            logger.error(f"Error: Not all expected files ({EXPECTED_RAW_FILES}) were found in {raw_dir} after download attempt.")
        else:
            logger.info(f"Successfully found expected raw files in {raw_dir}")
            
            preprocess_text_files_for_causal_lm(
                str(raw_dir), str(processed_dir), tokenizer_name_for_example, example_config
            )

            logger.info("Successfully finished preprocessing all splits with line-by-line processing.")
            logger.info(f"Please manually check the output .jsonl files in {processed_dir} "
                        "to ensure correct format (single JSON line, 'input_ids' and 'attention_mask' as flat lists).")

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
                    logger.info(f"\nCreated CausalLMTrainingDataset with {len(dataset)} examples.")
                    first_batch = next(iter(dataloader))
                    logger.info("\nFirst batch details:")
                    logger.info(f"  Input IDs shape: {first_batch['input_ids'].shape}")
                    logger.info(f"  Attention Mask shape: {first_batch['attention_mask'].shape}")
                    logger.info(f"  Labels shape: {first_batch['labels'].shape}")
                else:
                    logger.info("Dataset created but contains no examples (check log for token counts and sequence length).")
            else:
                logger.info(f"Processed training file {train_jsonl_path} not found.")

    except Exception as e:
        logger.error(f"An error occurred in the example usage: {e}", exc_info=True)
    finally:
        if base_dir.exists():
            # shutil.rmtree(base_dir) # Keeping for inspection
            logger.info(f"\nCleanup of {base_dir} skipped for inspection. Delete manually if needed.")
        else:
            logger.info(f"\nBase directory {base_dir} does not exist, no cleanup needed or already cleaned.")

    logger.info("\nText Dataset Utilities example finished.")