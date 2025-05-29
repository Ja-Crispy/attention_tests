import argparse
import torch
import random
import numpy as np
import logging
from pathlib import Path

# Project-specific imports
# Assuming this script is run as a module from the project root, e.g.,
# python -m my_attention_research_project.training.experiment_runner --config ...
try:
    from ..utils.config_parser import load_config
    from ..data.tokenizer import load_tokenizer
    from ..data.wikitext103 import download_raw_text_files, preprocess_text_files_for_causal_lm, CausalLMTrainingDataset
    from ..models.transformer import TransformerEncoder
    from .train_loop import Trainer
except ImportError as e:
    # Allow running script directly for development/testing if paths are adjusted or project is in PYTHONPATH
    print(f"ImportError: {e}. This might be due to running the script directly without the project package being installed or in PYTHONPATH.")
    print("Attempting to proceed. If project-specific imports fail, ensure you run as a module or adjust PYTHONPATH.")
    # Re-raise if it's not a direct run (__name__ != '__main__') because then it's a structural problem.
    if __name__ != '__main__':
        raise
    # For direct run, these imports might fail if the relative paths don't resolve.
    # This block is more for awareness during development.
    # A robust solution for direct script running might involve temporarily adding project root to sys.path.


# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Optional: for full determinism, might impact performance
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False   
    logger.info(f"Set random seed to {seed_value}")

def main():
    parser = argparse.ArgumentParser(description="Experiment Runner for Transformer Models")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load Configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    if not config:
        logger.error("Configuration file could not be loaded. Exiting.")
        return

    # Set Seed
    seed = config.get('training', {}).get('random_seed', 42)
    set_seed(seed)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Tokenizer Initialization
    tokenizer_name_or_path = config.get('data', {}).get('tokenizer_name_or_path', 'gpt2')
    tokenizer_max_len = config.get('data', {}).get('tokenizer_max_length', 1024) # Used by preprocess & load_tokenizer for context
    
    logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len)
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # load_tokenizer should set pad_token = eos_token if pad is missing (e.g. for GPT2)
        # So, pad_token_id should then be tokenizer.eos_token_id
        # If both are None, it's an issue.
        if tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Tokenizer's pad_token_id is None. Using eos_token_id ({pad_token_id}) as pad_token_id for CausalLMTrainingDataset.")
        else:
            # This should ideally not happen with standard Hugging Face tokenizers
            logger.error("Tokenizer does not have a pad_token_id or eos_token_id. Cannot proceed.")
            # Defaulting to 0, but this is risky and dataset/model might behave unexpectedly.
            pad_token_id = 0 
            logger.warning("Setting pad_token_id to 0 as a fallback. This is not recommended.")

    # Data Preparation
    # Using experiment-specific subdirectories for raw and processed data
    experiment_data_base_dir = config.get('data', {}).get('experiment_data_base_dir', './data/experiment_run_data')
    run_name = config.get('experiment_name', Path(args.config).stem) # Use config name as default run_name
    
    raw_data_dir = Path(experiment_data_base_dir) / run_name / 'raw'
    processed_data_dir = Path(experiment_data_base_dir) / run_name / 'processed'
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")

    # Ensure directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading/creating raw text files...")
    download_raw_text_files(str(raw_data_dir)) # Creates dummy train.txt, valid.txt, test.txt

    logger.info("Preprocessing text files for Causal LM...")
    # Pass the tokenizer_name_or_path, not the instance, as preprocess_text_files_for_causal_lm re-loads it.
    preprocess_text_files_for_causal_lm(str(raw_data_dir), str(processed_data_dir), tokenizer_name_or_path, config)

    train_file = processed_data_dir / "train.jsonl"
    valid_file = processed_data_dir / "valid.jsonl"
    # test_file = processed_data_dir / "test.jsonl" # For future use

    sequence_length = config.get('data', {}).get('sequence_length', 256)
    
    logger.info(f"Loading training dataset from: {train_file}")
    train_dataset = CausalLMTrainingDataset(str(train_file), sequence_length, pad_token_id)
    logger.info(f"Training dataset size: {len(train_dataset)} examples")
    
    logger.info(f"Loading validation dataset from: {valid_file}")
    valid_dataset = CausalLMTrainingDataset(str(valid_file), sequence_length, pad_token_id)
    logger.info(f"Validation dataset size: {len(valid_dataset)} examples")

    # Model Initialization
    model_config = config.get('model', {})
    vocab_size = tokenizer.vocab_size # Use actual vocab size from the loaded tokenizer
    logger.info(f"Initializing model (TransformerEncoder) with vocab_size: {vocab_size}")

    # Extract attention-specific configuration from the model_config
    attention_specific_config = model_config.get('attention')
    if not attention_specific_config:
        logger.error("Key 'attention' missing in model configuration ('model.attention'). Cannot initialize TransformerEncoder.")
        return # Or raise a more specific error

    # TODO: Add logic to select model type based on config['model']['type'] if more models are introduced.
    # For now, directly uses TransformerEncoder.
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 256),
        attention_config=attention_specific_config, # Pass the extracted attention configuration
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        ffn_dim_factor=model_config.get('ffn_dim_factor', 4),
        dropout_rate=model_config.get('dropout_rate', 0.1),
        use_positional_embeddings=model_config.get('use_positional_embeddings', True),
        pos_encoding_max_len=model_config.get('pos_encoding_max_len', 5000),
        embedding_scale_factor=model_config.get('embedding_scale_factor', None) 
        # embedding_scale_grad_by_freq is False by default in TransformerEncoder
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized. Total trainable parameters: {num_params/1e6:.2f}M")

    # Trainer Initialization
    logger.info("Initializing Trainer...")
    # The config passed to Trainer should ideally be the full config dict for flexibility
    trainer = Trainer(config, model, train_dataset, valid_dataset)

    # Start Training
    logger.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
    finally:
        logger.info("Training finished or an error occurred.")
        # Optional: cleanup experiment data directories if they were temporary
        # cleanup_data = config.get('training', {}).get('cleanup_experiment_data_on_finish', False)
        # if cleanup_data:
        #     logger.info(f"Cleaning up experiment data directory: {Path(experiment_data_base_dir) / run_name}")
        #     shutil.rmtree(Path(experiment_data_base_dir) / run_name)


if __name__ == '__main__':
    # To run this script:
    # 1. Ensure your project root is in PYTHONPATH or you are in the project root.
    # 2. Execute as a module:
    #    python -m my_attention_research_project.training.experiment_runner --config my_attention_research_project/configs/base_mha_config.yaml
    #    (Adjust path to config as needed)
    #
    # If running directly for quick tests and imports fail, you might need to adjust sys.path:
    # import sys
    # sys.path.insert(0, str(Path(__file__).resolve().parents[2])) # Add project root to path
    # from my_attention_research_project.utils.config_parser import load_config
    # ... etc.
    # However, running as a module is preferred.
    main()
