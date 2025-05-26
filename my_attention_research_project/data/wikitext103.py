# text_dataset_utils.py (conceptually, actual filename: wikitext103.py)
# Handles downloading/creating raw text, preprocessing for Causal LM, and serving via PyTorch Dataset.

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import shutil # For robust directory removal in examples
import json # For saving/loading processed data list

# Import from the project's tokenizer.py
from .tokenizer import load_tokenizer, tokenize_function 

def download_raw_text_files(raw_data_dir: str) -> None:
    """
    Creates dummy raw text files (train.txt, valid.txt, test.txt) in the specified directory.
    This simulates a dataset download for easier testing.

    Args:
        raw_data_dir (str): Path to the directory where raw text files will be created.
    """
    print(f"Creating dummy raw text files in {raw_data_dir}...")
    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)

    dummy_train_text = (
        "Chapter 1: The Beginning.\nIt was a dark and stormy night. The wind howled, and rain lashed against the window panes. "
        "Inside, by the fireplace, sat a lone figure, pondering the mysteries of the universe. "
        "This text is meant for training. It needs to be long enough to create multiple sequences. "
        "Causal language modeling is a fascinating subject. We predict the next token given previous tokens. "
        "Let's add more sentences to make this sufficiently long for our dummy dataset. "
        + ("This is a repeated sentence for length. " * 100)
        + "End of training chapter 1."
    )
    (Path(raw_data_dir) / "train.txt").write_text(dummy_train_text, encoding='utf-8')

    dummy_valid_text = (
        "Chapter 1: Validation Passages.\nValidation text helps tune hyperparameters. It should be different from the training set. "
        "Consider this a new chapter, exploring similar themes but with different words. "
        "The figure by the fireplace stirred, a new thought dawning. "
        + ("This is a repeated validation sentence. " * 50)
        + "End of validation chapter 1."
    )
    (Path(raw_data_dir) / "valid.txt").write_text(dummy_valid_text, encoding='utf-8')

    dummy_test_text = (
        "Chapter 1: The Final Test.\nTest data provides the final evaluation. It must not be seen during training or validation. "
        "The storm had passed, and a quiet dawn approached. The answers were becoming clear. "
        + ("This is a repeated test sentence. " * 50)
        + "End of test chapter 1."
    )
    (Path(raw_data_dir) / "test.txt").write_text(dummy_test_text, encoding='utf-8')
    
    print(f"Created dummy files: train.txt, valid.txt, test.txt in {raw_data_dir}")

def preprocess_text_files_for_causal_lm(
    raw_data_dir: str, 
    processed_data_dir: str, 
    tokenizer_name_or_path: str, 
    config: dict
) -> None:
    """
    Processes raw text files for Causal Language Modeling.
    Each raw text file (train.txt, valid.txt, test.txt) is tokenized
    and its entire content is saved as a single JSON line in a corresponding .jsonl file.

    Args:
        raw_data_dir (str): Directory containing raw text files (train.txt, etc.).
        processed_data_dir (str): Directory to save the processed .jsonl files.
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer.
        config (dict): Configuration dictionary, expected to contain:
                       `config['data']['tokenizer_max_length']`
                       `config['data']['sequence_length']` (though not directly used here, good for context)
    """
    print(f"Preprocessing text files from {raw_data_dir} to {processed_data_dir} for Causal LM...")
    Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_max_len = config.get('data', {}).get('tokenizer_max_length', 1024)
    print(f"Using tokenizer: {tokenizer_name_or_path} with max_length: {tokenizer_max_len}")
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len)

    for split in ["train", "valid", "test"]:
        raw_file_path = Path(raw_data_dir) / f"{split}.txt"
        processed_file_path = Path(processed_data_dir) / f"{split}.jsonl"

        if not raw_file_path.exists():
            print(f"Warning: Raw file {raw_file_path} not found. Skipping preprocessing for this split.")
            continue

        print(f"Processing {raw_file_path}...")
        text_content = raw_file_path.read_text(encoding='utf-8')

        if not text_content.strip():
            print(f"Warning: Raw file {raw_file_path} is empty or contains only whitespace. Skipping.")
            continue
        
        # Tokenize the entire content of the file.
        # Using do_not_pad as we are tokenizing the whole file, then chunking in Dataset.
        # Truncation is important if file content > tokenizer_max_len.
        tokenized_data = tokenize_function(
            text_content, 
            tokenizer, 
            max_length=tokenizer_max_len, # This will truncate if file content is too long
            padding_strategy="do_not_pad", # No padding, we get one long sequence
            truncation_strategy=True
        )
        
        input_ids_list = tokenized_data["input_ids"].tolist()
        attention_mask_list = tokenized_data["attention_mask"].tolist()

        # Save as a single JSON line in the .jsonl file
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json_record = {"input_ids": input_ids_list, "attention_mask": attention_mask_list}
            f.write(json.dumps(json_record) + '\n')
        
        print(f"Saved processed data for {split} to {processed_file_path} ({len(input_ids_list)} tokens).")

    print("Preprocessing complete.")


class CausalLMTrainingDataset(Dataset):
    def __init__(self, file_path: str, sequence_length: int, pad_token_id: int):
        """
        Dataset for Causal Language Modeling.
        Reads a .jsonl file where each line contains tokenized 'input_ids' and 'attention_mask'
        for an entire dataset split (e.g., all training text).
        This Dataset class then chunks this long sequence into smaller, fixed-length sequences.

        Args:
            file_path (str): Path to the processed .jsonl file (e.g., train.jsonl).
                             Expects a single JSON line in the file.
            sequence_length (int): The length of sequences to return.
            pad_token_id (int): The ID of the PAD token from the tokenizer, used for padding if needed
                                (though not explicitly used for padding here, good for context).
        """
        print(f"Initializing CausalLMTrainingDataset from: {file_path}, sequence_length: {sequence_length}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline() # Expecting only one line for the entire split
            if not line:
                raise ValueError(f"Processed data file {file_path} is empty.")
            try:
                data_record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON from {file_path}. Ensure it's a valid JSON line.")

        self.input_ids = torch.tensor(data_record['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(data_record['attention_mask'], dtype=torch.long)
        
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id # Stored for reference, might be useful later

        if self.input_ids.dim() != 1 or self.attention_mask.dim() != 1:
            raise ValueError("Loaded 'input_ids' and 'attention_mask' must be 1D tensors (lists in JSON).")
        if self.input_ids.size(0) != self.attention_mask.size(0):
            raise ValueError("'input_ids' and 'attention_mask' must have the same length.")

        # Ensure at least one example by padding if necessary
        min_tokens = self.sequence_length + 1
        if self.input_ids.size(0) < min_tokens:
            pad_len = min_tokens - self.input_ids.size(0)
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros((pad_len,), dtype=torch.long)
            self.input_ids = torch.cat([self.input_ids, pad_ids], dim=0)
            self.attention_mask = torch.cat([self.attention_mask, pad_attn], dim=0)
            print(f"Padded dataset to {min_tokens} tokens for sequence length {self.sequence_length}.")

        # Calculate the number of examples. We drop any extra tokens beyond full sequences.
        # Each example is sequence_length tokens input and sequence_length tokens target (shifted by 1).
        self.num_examples = (self.input_ids.size(0) - 1) // self.sequence_length
        print(f"Dataset loaded. Total tokens: {self.input_ids.size(0)}. Num examples for seq_len {self.sequence_length}: {self.num_examples}")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        Returns a dictionary `{"input_ids": ..., "attention_mask": ..., "labels": ...}`.
        'labels' are the target_ids, with padding tokens masked to -100.
        """
        start_index = idx * self.sequence_length
        # Slice to get sequence_length + 1 tokens to form input and target
        # This chunk contains tokens for both input and the target (shifted by one)
        end_index = start_index + self.sequence_length + 1
        full_chunk = self.input_ids[start_index:end_index]
        
        input_ids_chunk = full_chunk[:-1]
        target_ids_chunk = full_chunk[1:].clone() # Clone to modify for labels

        # Fetch the corresponding attention mask for the input sequence
        # This mask indicates real tokens (1) vs padding (0) in the original tokenized stream.
        # Note: If preprocess used do_not_pad and truncate, attention_mask for the loaded part should be all 1s
        # up to the truncation length of the original file.
        attention_mask_chunk = self.attention_mask[start_index : start_index + self.sequence_length]

        # Create labels: where attention_mask_chunk is 0 (padding in source), set target_ids to -100
        # This is crucial for Causal LM if the original long sequence had padding due to truncation
        # by tokenizer_max_length when the whole file was tokenized.
        # If using 'do_not_pad' and the source text was shorter than tokenizer_max_length,
        # the attention_mask for actual tokens will be all 1s.
        target_ids_chunk.masked_fill_(attention_mask_chunk == 0, -100)
        
        return {
            "input_ids": input_ids_chunk, 
            "attention_mask": attention_mask_chunk, 
            "labels": target_ids_chunk
        }

if __name__ == '__main__':
    print("Text Dataset Utilities example usage:")
    
    main_example_dir_name = "text_dataset_utils_example_run"
    base_dir = Path("./data") / main_example_dir_name
    raw_dir = base_dir / "raw_text"
    processed_dir = base_dir / "processed_text_for_causal_lm"
    
    # Ensure a clean state for the example run
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Config for the example
    # For GPT-2, pad_token_id is often the same as eos_token_id (e.g., 50256)
    # We will fetch it from the tokenizer after loading.
    example_config = {
        'data': {
            'sequence_length': 64, 
            'tokenizer_max_length': 256 # Smaller for faster example processing of dummy text
        },
        'model': { # This section might be for model parameters, but we use pad_token_id from tokenizer
            # 'pad_token_id': 50256 # Placeholder, will be set by tokenizer
        }
    }
    tokenizer_name_for_example = "gpt2" 

    try:
        # 1. "Download" raw text files (creates dummy files)
        download_raw_text_files(str(raw_dir))

        # 2. Preprocess text files
        preprocess_text_files_for_causal_lm(
            raw_data_dir=str(raw_dir),
            processed_data_dir=str(processed_dir),
            tokenizer_name_or_path=tokenizer_name_for_example,
            config=example_config
        )

        # 3. Instantiate Dataset and DataLoader
        # Load tokenizer once to get pad_token_id for the Dataset
        # This ensures pad_token_id is consistent with the tokenizer used for preprocessing.
        temp_tokenizer = load_tokenizer(tokenizer_name_for_example)
        pad_token_id_for_dataset = temp_tokenizer.pad_token_id
        if pad_token_id_for_dataset is None: # Should be handled by load_tokenizer, but as a safeguard
            pad_token_id_for_dataset = temp_tokenizer.eos_token_id if temp_tokenizer.eos_token_id is not None else 0
            print(f"Warning: PAD token ID was None, using EOS or 0: {pad_token_id_for_dataset}")
        del temp_tokenizer # No longer needed

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
                print(f"  Input IDs shape: {first_batch['input_ids'].shape}")   # Expected: [batch_size, sequence_length]
                print(f"  Attention Mask shape: {first_batch['attention_mask'].shape}") # Expected: [batch_size, sequence_length]
                print(f"  Labels shape: {first_batch['labels'].shape}")         # Expected: [batch_size, sequence_length]
                
                print("\nSample from first batch:")
                print(f"  Input IDs (sample 0): {first_batch['input_ids'][0][:20]}...") # Print first 20 tokens
                print(f"  Attention Mask (sample 0): {first_batch['attention_mask'][0][:20]}...")
                print(f"  Labels (sample 0): {first_batch['labels'][0][:20]}...")
                # Check if -100 is present in labels where attention mask might be 0 (if any padding occurred)
                # For this dummy data and 'do_not_pad' with sufficient tokenizer_max_length, mask should be all 1s.
                if (first_batch['labels'] == -100).any():
                    print("  Note: -100 found in labels, indicating masked tokens.")
                else:
                    print("  Note: No -100 found in labels for this batch (likely no padding in source chunk or mask is all 1s).")

            else:
                print("Dataset created but contains no examples. Dummy data might be too short or seq_len too long.")
        else:
            print(f"Processed training file {train_jsonl_path} not found. Cannot create dataset.")

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files and directories
        if base_dir.exists():
            shutil.rmtree(base_dir)
            print(f"\nCleaned up example directory: {base_dir}")

    print("\nText Dataset Utilities example finished.")
