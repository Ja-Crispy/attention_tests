import argparse
import torch
import random
import numpy as np
import logging
from pathlib import Path
import os # Added
import pandas as pd # Added

# Project-specific imports
try:
    from ..utils.config_parser import load_config
    from ..data.tokenizer import load_tokenizer
    from ..data.wikitext103 import download_raw_text_files, preprocess_text_files_for_causal_lm, CausalLMTrainingDataset
    from ..models.transformer import TransformerEncoder
    from .train_loop import Trainer
except ImportError as e:
    print(f"ImportError: {e}. This might be due to running the script directly without the project package being installed or in PYTHONPATH.")
    print("Attempting to proceed. If project-specific imports fail, ensure you run as a module or adjust PYTHONPATH.")
    if __name__ != '__main__':
        raise

# Logger Setup
logger = logging.getLogger(__name__)
# Ensure basicConfig is called only if no handlers are already configured
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False   
    logger.info(f"Set random seed to {seed_value}")

def main(): # This function serves as run_experiment
    parser = argparse.ArgumentParser(description="Experiment Runner for Transformer Models")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    config_path = args.config # Store config path for logging

    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    if not config:
        logger.error("Configuration file could not be loaded. Exiting.")
        return

    seed = config.get('training', {}).get('random_seed', 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer_name_or_path = config.get('data', {}).get('tokenizer_name_or_path', 'gpt2')
    tokenizer_max_len = config.get('data', {}).get('tokenizer_max_length', 1024)
    
    logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = load_tokenizer(tokenizer_name_or_path, max_length=tokenizer_max_len)
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Tokenizer's pad_token_id is None. Using eos_token_id ({pad_token_id}) as pad_token_id.")
        else:
            pad_token_id = 0 
            logger.warning("Tokenizer does not have pad_token_id or eos_token_id. Setting pad_token_id to 0.")

    experiment_data_base_dir = config.get('data', {}).get('experiment_data_base_dir', './data/experiment_run_data')
    run_name = config.get('experiment_name', Path(config_path).stem) 
    
    raw_data_dir = Path(experiment_data_base_dir) / run_name / 'raw'
    processed_data_dir = Path(experiment_data_base_dir) / run_name / 'processed'
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading/creating raw text files...")
    download_raw_text_files(str(raw_data_dir))

    # Check if preprocessing can be skipped
    processed_files = ["train.jsonl", "valid.jsonl", "test.jsonl"]
    all_processed_exist = all((processed_data_dir / pf).exists() for pf in processed_files)
    
    if all_processed_exist:
        # Check if processed files are newer than raw files
        raw_files = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]
        newest_raw_time = max((raw_data_dir / rf).stat().st_mtime for rf in raw_files if (raw_data_dir / rf).exists())
        oldest_processed_time = min((processed_data_dir / pf).stat().st_mtime for pf in processed_files)
        
        if oldest_processed_time > newest_raw_time:
            logger.info("Processed files are up-to-date. Skipping preprocessing...")
            skip_preprocessing = True
        else:
            logger.info("Raw files are newer than processed files. Re-preprocessing...")
            skip_preprocessing = False
    else:
        logger.info("Some processed files are missing. Preprocessing required...")
        skip_preprocessing = False

    if not skip_preprocessing:
        logger.info("Preprocessing text files for Causal LM...")
        preprocess_text_files_for_causal_lm(str(raw_data_dir), str(processed_data_dir), tokenizer_name_or_path, config)

    train_file = processed_data_dir / "train.jsonl"
    valid_file = processed_data_dir / "valid.jsonl"
    sequence_length = config.get('data', {}).get('sequence_length', 256)
    
    logger.info(f"Loading training dataset from: {train_file}")
    train_dataset = CausalLMTrainingDataset(str(train_file), sequence_length, pad_token_id)
    logger.info(f"Training dataset size: {len(train_dataset)} examples")
    
    logger.info(f"Loading validation dataset from: {valid_file}")
    valid_dataset = CausalLMTrainingDataset(str(valid_file), sequence_length, pad_token_id)
    logger.info(f"Validation dataset size: {len(valid_dataset)} examples")

    model_config = config.get('model', {})
    vocab_size = tokenizer.vocab_size
    logger.info(f"Initializing model (TransformerEncoder) with vocab_size: {vocab_size}")

    attention_specific_config = model_config.get('attention')
    if not attention_specific_config:
        logger.error("Key 'attention' missing in model configuration. Cannot initialize TransformerEncoder.")
        return

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 256),
        attention_config=attention_specific_config,
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        ffn_dim_factor=model_config.get('ffn_dim_factor', 4),
        dropout_rate=model_config.get('dropout_rate', 0.1),
        use_positional_embeddings=model_config.get('use_positional_embeddings', True),
        pos_encoding_max_len=model_config.get('pos_encoding_max_len', 5000),
        embedding_scale_factor=model_config.get('embedding_scale_factor', None) 
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized. Total trainable parameters: {num_params/1e6:.2f}M")

    logger.info("Initializing Trainer...")
    trainer = Trainer(config, model, train_dataset, valid_dataset)

    logger.info("Starting training...")
    all_epoch_summaries = [] # Initialize in case training fails early
    try:
        all_epoch_summaries = trainer.train() # Capture the returned list of metrics
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
    finally:
        logger.info("Training finished or an error occurred.")
        
        if all_epoch_summaries:
            df_metrics = pd.DataFrame(all_epoch_summaries)
            
            # Add configuration parameters
            df_metrics['config_file'] = Path(config_path).name # Log only filename for brevity
            
            # Safely access nested config values
            cfg_model = config.get('model', {})
            cfg_model_attention = cfg_model.get('attention', {})
            cfg_data = config.get('data', {})

            df_metrics['attention_type'] = cfg_model_attention.get('type', 'N/A')
            df_metrics['d_model'] = cfg_model.get('d_model', 'N/A')
            df_metrics['num_encoder_layers'] = cfg_model.get('num_encoder_layers', 'N/A')
            df_metrics['sequence_length'] = cfg_data.get('sequence_length', 'N/A')

            attention_type = cfg_model_attention.get('type')
            if attention_type == "AHC":
                df_metrics['ahc_chunk_size'] = cfg_model_attention.get('chunk_size', 'N/A')
            elif attention_type == "PAS":
                scanner_cfg = cfg_model_attention.get('scanner_config', {})
                df_metrics['pas_top_k_hotspots'] = scanner_cfg.get('top_k_hotspots', 'N/A')
            elif attention_type == "MOAE":
                expert_cfgs = cfg_model_attention.get('expert_configs', [])
                df_metrics['moae_num_experts'] = len(expert_cfgs) if isinstance(expert_cfgs, list) else 'N/A'
            
            # Define CSV output path
            output_dir = Path(config.get('training', {}).get('results_dir', './experiment_results/'))
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_output_path = output_dir / "benchmark_results.csv"

            # Save to CSV
            try:
                if os.path.exists(csv_output_path):
                    df_metrics.to_csv(csv_output_path, mode='a', header=False, index=False)
                    logger.info(f"Appended benchmark results to {csv_output_path}")
                else:
                    df_metrics.to_csv(csv_output_path, mode='w', header=True, index=False)
                    logger.info(f"Saved new benchmark results to {csv_output_path}")
            except Exception as e_csv:
                logger.error(f"Error saving results to CSV {csv_output_path}: {e_csv}", exc_info=True)
        else:
            logger.warning("No epoch summaries were collected. Skipping CSV logging.")


if __name__ == '__main__':
    main()
