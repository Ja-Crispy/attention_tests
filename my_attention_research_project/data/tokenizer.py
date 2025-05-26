# tokenizer.py
# Handles tokenizer loading (primarily for pre-trained Hugging Face tokenizers)
# and provides utilities for tokenizing text.
# Also includes functions for training a custom ByteLevelBPETokenizer from scratch.

from tokenizers import ByteLevelBPETokenizer # For custom tokenizer training
from transformers import AutoTokenizer
from pathlib import Path
import shutil # For robust directory removal in examples
import torch # For tokenize_function and __main__ example

# --- Functions for Training a Custom ByteLevelBPETokenizer ---

def initialize_tokenizer(vocab_size: int = 30000, min_frequency: int = 2) -> ByteLevelBPETokenizer:
    """
    Initializes a new ByteLevelBPETokenizer for training a custom tokenizer from scratch.
    This is typically not used if you plan to use a pre-trained Hugging Face tokenizer.

    Args:
        vocab_size (int): The desired vocabulary size.
        min_frequency (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        ByteLevelBPETokenizer: An initialized (but not trained) tokenizer instance.
    """
    print(f"Initializing NEW ByteLevelBPETokenizer for custom training (vocab_size={vocab_size}, min_frequency={min_frequency})")
    tokenizer = ByteLevelBPETokenizer()
    # In a real scenario for custom training, you might add more special tokens or configurations here.
    return tokenizer

def train_tokenizer(
    tokenizer: ByteLevelBPETokenizer,
    file_paths: list[str], # List of paths to raw text files for training
    vocab_size: int = 30000,
    min_frequency: int = 2,
    output_path: str = "./data/custom_tokenizers", # Directory to save custom tokenizer
    tokenizer_name: str = "custom_bpe_tokenizer"
) -> None:
    """
    Trains the given ByteLevelBPETokenizer instance on the provided text files and saves it.
    This is for training a custom tokenizer from scratch. For using pre-trained models,
    refer to `load_tokenizer`.

    Args:
        tokenizer (ByteLevelBPETokenizer): The tokenizer instance to train.
        file_paths (list[str]): List of paths to raw text files for training.
        vocab_size (int): Target vocabulary size.
        min_frequency (int): Minimum token frequency.
        output_path (str): Directory to save the trained tokenizer files.
        tokenizer_name (str): Name for the tokenizer files (e.g., 'my_custom_bpe').
    """
    print(f"Starting custom training for ByteLevelBPETokenizer on files: {file_paths}...")
    # Placeholder: Actual training logic for ByteLevelBPETokenizer
    # tokenizer.train(files=file_paths, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
    #     "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    # ])
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Custom tokenizers often save vocab.json and merges.txt separately
    save_files_prefix = Path(output_path) / tokenizer_name
    # tokenizer.save_model(str(output_path), tokenizer_name) # Correct would be tokenizer.save_model(output_path, tokenizer_name)
                                                        # This creates tokenizer_name-vocab.json and tokenizer_name-merges.txt

    # For this placeholder, we'll just touch the files to simulate creation
    (Path(output_path) / f"{tokenizer_name}-vocab.json").touch() 
    (Path(output_path) / f"{tokenizer_name}-merges.txt").touch() 
    print(f"Placeholder: Custom Tokenizer training complete. Would save to {save_files_prefix}-vocab.json and {save_files_prefix}-merges.txt")
    print("NOTE: Actual custom tokenizer training is a more involved process and is stubbed here.")

# --- Functions for Loading and Using Pre-trained Hugging Face Tokenizers ---

def load_tokenizer(tokenizer_name_or_path: str, max_length: int = None):
    """
    Loads a pre-trained tokenizer from Hugging Face Model Hub or a local path
    using AutoTokenizer.

    Args:
        tokenizer_name_or_path (str): The name of the pre-trained model (e.g., "bert-base-uncased")
                                      or path to a local directory containing tokenizer files.
        max_length (int, optional): If provided, this information can be used to set the
                                   tokenizer's model_max_length if not already set or if the
                                   new value is smaller. Actual truncation/padding behavior is
                                   controlled by the `tokenize_function`. Defaults to None.
    Returns:
        A Hugging Face tokenizer instance (e.g., BertTokenizerFast, GPT2TokenizerFast).
    Raises:
        OSError: If the tokenizer cannot be loaded (e.g., network issue, model not found).
    """
    print(f"Loading Hugging Face tokenizer: {tokenizer_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    except OSError as e:
        print(f"Error loading tokenizer {tokenizer_name_or_path}: {e}")
        print("Please ensure the tokenizer name is correct, you have internet connectivity (if loading from hub),")
        print("or the local path is valid and contains the necessary tokenizer files.")
        raise

    if max_length:
        # tokenizer.model_max_length is often a very large number or None for some models initially.
        # Setting it here is more for consistency if a project assumes a max_length.
        # The actual sequence length for model input is determined during tokenization by tokenize_function.
        if tokenizer.model_max_length is None or tokenizer.model_max_length > max_length:
            # Some tokenizers might issue warnings if you directly set model_max_length if it's already defined.
            # However, for many use cases, it's acceptable if done carefully.
            # tokenizer.model_max_length = max_length # This line can be uncommented if direct override is desired.
            print(f"Note: Requested max_length for tokenizer: {max_length}. The tokenizer's default max length is {tokenizer.model_max_length}.")
            print("Actual truncation/padding to `max_length` is handled in `tokenize_function`.")
        else:
            print(f"Tokenizer's existing model_max_length ({tokenizer.model_max_length}) is already less than or equal to requested {max_length}.")


    # Handle PAD token for models that might not have it pre-defined (e.g., GPT-2)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print(f"Tokenizer missing PAD token. Using EOS token ('{tokenizer.eos_token}') as PAD token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # This case is less common for standard pre-trained models
            new_pad_token = '[PAD]'
            print(f"Tokenizer missing PAD and EOS token. Adding a new PAD token: {new_pad_token}")
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
    
    print(f"Tokenizer '{tokenizer_name_or_path}' loaded successfully.")
    print(f"  Class: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  BOS/CLS token: '{tokenizer.bos_token or tokenizer.cls_token}' (ID: {tokenizer.bos_token_id or tokenizer.cls_token_id})")
    print(f"  Effective model_max_length for truncation/padding (if set): {tokenizer.model_max_length}")

    return tokenizer

def tokenize_function(text: str, tokenizer, max_length: int, padding_strategy: str = "max_length", truncation_strategy: bool = True) -> dict:
    """
    Tokenizes a single string of text using the provided Hugging Face tokenizer.

    Args:
        text (str): The input text string.
        tokenizer: An instance of a loaded Hugging Face tokenizer.
        max_length (int): The maximum sequence length for padding/truncation.
        padding_strategy (str, optional): Padding strategy. Defaults to "max_length".
                                          Options: "max_length", "do_not_pad", "longest".
        truncation_strategy (bool, optional): Whether to truncate sequences longer than max_length.
                                             Defaults to True.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' as PyTorch tensors.
              Tensors are squeezed to 1D (shape [seq_len]) as this function processes a single text string.
    """
    if not text: # Handle empty string case
        print("Warning: Received empty string for tokenization.")
        empty_tensor = torch.tensor([], dtype=torch.long)
        return {'input_ids': empty_tensor, 'attention_mask': empty_tensor}

    tokenized_output = tokenizer(
        text,
        max_length=max_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
        return_attention_mask=True
    )
    # Squeeze to remove the batch dimension, as this function handles a single text string.
    return {
        "input_ids": tokenized_output["input_ids"].squeeze(0),
        "attention_mask": tokenized_output["attention_mask"].squeeze(0)
    }

if __name__ == '__main__':
    print("Tokenizer module example usage:\n")

    # --- Example for loading and using a Hugging Face pre-trained tokenizer ---
    print("--- Hugging Face AutoTokenizer Example ---")
    # Using a smaller model for quicker download in example, e.g., "distilbert-base-uncased" or "gpt2"
    # For GPT2, pad_token will be set to eos_token.
    # For bert-base-uncased, it already has a pad_token.
    tokenizer_name = "gpt2" # "distilbert-base-uncased" is another good small option
    example_max_len = 64
    
    try:
        hf_tokenizer = load_tokenizer(tokenizer_name, max_length=example_max_len)
        
        sample_text = "Hello, world! This is an example sentence for the Hugging Face tokenizer."
        print(f"\nTokenizing sample text: \"{sample_text}\" with max_length={example_max_len}")
        
        tokenized_result = tokenize_function(sample_text, hf_tokenizer, max_length=example_max_len)
        
        print("\nTokenized Result:")
        print(f"  Input IDs: {tokenized_result['input_ids']}")
        print(f"  Input IDs Shape: {tokenized_result['input_ids'].shape}")
        print(f"  Attention Mask: {tokenized_result['attention_mask']}")
        print(f"  Attention Mask Shape: {tokenized_result['attention_mask'].shape}")
        
        decoded_text = hf_tokenizer.decode(tokenized_result['input_ids'])
        print(f"  Decoded Input IDs: \"{decoded_text}\"")

    except Exception as e:
        print(f"Error in Hugging Face tokenizer example: {e}")
        print("This example might fail if you are offline or the model name is incorrect.")

    print("\n--- End of Hugging Face AutoTokenizer Example ---\n")

    # --- Example for training a custom ByteLevelBPETokenizer (Placeholder) ---
    print("--- Custom ByteLevelBPETokenizer Training Example (Placeholder) ---")
    main_example_dir_name = "custom_tokenizer_example_run"
    base_data_dir = Path("./data") 
    
    example_base_dir = base_data_dir / main_example_dir_name
    dummy_text_files_dir = example_base_dir / "dummy_text_for_custom_tokenizer"
    custom_tokenizer_output_dir = example_base_dir / "custom_tokenizers_output"

    if example_base_dir.exists():
        shutil.rmtree(example_base_dir)
    dummy_text_files_dir.mkdir(parents=True, exist_ok=True)
    custom_tokenizer_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        custom_bpe_tokenizer = initialize_tokenizer(vocab_size=1000, min_frequency=1) # Smaller for example
        
        dummy_files_for_custom_training = [str(dummy_text_files_dir / "dummy_text1.txt"), str(dummy_text_files_dir / "dummy_text2.txt")]
        Path(dummy_files_for_custom_training[0]).write_text("This is sample sentence one for custom training.")
        Path(dummy_files_for_custom_training[1]).write_text("Another sample sentence, sentence two, for custom BPE tokenizer.")
        
        train_tokenizer(
            custom_bpe_tokenizer,
            dummy_files_for_custom_training,
            vocab_size=1000,
            min_frequency=1,
            output_path=str(custom_tokenizer_output_dir),
            tokenizer_name="my_custom_bpe"
        )
        print("Custom tokenizer training placeholder executed.")
        # Placeholder for loading/testing the custom trained tokenizer would go here.
        # e.g., custom_loaded = ByteLevelBPETokenizer.from_file(
        #     vocab=str(custom_tokenizer_output_dir / "my_custom_bpe-vocab.json"),
        #     merges=str(custom_tokenizer_output_dir / "my_custom_bpe-merges.txt")
        # )
        # print("Custom tokenizer placeholder loading successful (simulated).")

    except Exception as e:
        print(f"Error in custom tokenizer training example: {e}")
    finally:
        if example_base_dir.exists():
            shutil.rmtree(example_base_dir)
            print(f"Cleaned up custom tokenizer example directory: {example_base_dir}")

    print("\n--- End of Custom ByteLevelBPETokenizer Training Example ---")
    print("\nTokenizer module example usage finished.")
