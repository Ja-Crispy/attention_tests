# train_loop.py
# Contains the main training loop and utilities.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import yaml # For config type hinting if used, or general dict
from pathlib import Path
import logging # For logging
import shutil # For __main__ example cleanup

# For type hinting if needed, or __main__ example.
# Ensure this path is correct based on your project structure.
# If train_loop.py is run directly, this might cause issues if not handled by try-except or if __main__ is complex.
try:
    from ..models.transformer import TransformerEncoder 
except ImportError:
    # This allows the script to run for the __main__ example even if models aren't perfectly on PYTHONPATH
    # when run directly, assuming DummyModel is self-contained for the example.
    if __name__ != '__main__': # Re-raise if not in main, as it's a real import error for library use
        raise
    print("Warning: Could not import TransformerEncoder. This is fine if running the __main__ example with DummyModel.")
    TransformerEncoder = None # Define it as None for type hinting if import fails

# Basic logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(self, config: dict, model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset = None):
        """
        Initializes the Trainer.

        Args:
            config (dict): Configuration dictionary.
            model (nn.Module): The PyTorch model to train.
            train_dataset (Dataset): Training dataset.
            valid_dataset (Dataset, optional): Validation dataset. Defaults to None.
        """
        logger.info("Initializing Trainer...")
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Optimizer
        optimizer_name = self.config.get('training', {}).get('optimizer', 'AdamW')
        lr = float(self.config.get('training', {}).get('learning_rate', 1e-4))
        weight_decay = float(self.config.get('training', {}).get('weight_decay', 0.01))

        if optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            logger.warning(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        logger.info(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")

        # Loss Function (CrossEntropyLoss for Language Modeling, ignoring -100 for labels)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # DataLoaders
        batch_size = self.config.get('data', {}).get('batch_size', 32)
        num_workers = self.config.get('data', {}).get('num_workers', 0) # Default to 0 for simpler setups
        pin_memory = self.config.get('data', {}).get('pin_memory', True if self.device.type == 'cuda' else False)

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        if self.valid_dataset:
            eval_batch_size = self.config.get('benchmarking', {}).get('batch_size_inference', batch_size)
            self.valid_loader = DataLoader(
                self.valid_dataset, 
                batch_size=eval_batch_size, 
                shuffle=False, 
                num_workers=num_workers, 
                pin_memory=pin_memory
            )
        else:
            self.valid_loader = None
            
        logger.info(f"Train DataLoader: batch_size={batch_size}, num_examples={len(self.train_dataset)}, num_workers={num_workers}")
        if self.valid_loader:
             logger.info(f"Valid DataLoader: batch_size={eval_batch_size}, num_examples={len(self.valid_dataset)}, num_workers={num_workers}")

        # LR Scheduler
        self.lr_scheduler = None
        scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
        if scheduler_config and scheduler_config.get('type'):
            scheduler_type = scheduler_config.get('type').lower()
            if scheduler_type == 'cosine_annealing':
                num_epochs = self.config.get('training', {}).get('num_epochs', 1)
                total_steps = num_epochs * len(self.train_loader)
                eta_min = float(scheduler_config.get('min_lr', 0.0))
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=total_steps,
                    eta_min=eta_min
                )
                logger.info(f"Using CosineAnnealingLR scheduler. T_max={total_steps}, eta_min={eta_min}")
            elif scheduler_type == 'linear_warmup_decay':
                # Placeholder - requires more parameters like warmup_steps
                logger.warning("LinearWarmupDecay scheduler placeholder - not implemented yet.")
            elif scheduler_type == 'reduce_lr_on_plateau':
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=scheduler_config.get('mode', 'min'),
                    factor=float(scheduler_config.get('factor', 0.1)),
                    patience=int(scheduler_config.get('patience', 10)),
                )
                logger.info(f"Using ReduceLROnPlateau scheduler.")
            else:
                logger.info(f"LR scheduler type '{scheduler_type}' not recognized or no LR scheduler configured.")
        else:
            logger.info("No LR scheduler configured.")

        self.gradient_clipping_norm = self.config.get('training', {}).get('gradient_clipping', None)
        if self.gradient_clipping_norm:
            logger.info(f"Gradient clipping enabled with max_norm: {self.gradient_clipping_norm}")
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')


    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        logger.info(f"Starting Training Epoch {self.current_epoch + 1}")
        epoch_start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            input_ids = batch_data["input_ids"].to(self.device)
            attention_mask = batch_data["attention_mask"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # logits shape: (batch_size, seq_len, vocab_size)
            # labels shape: (batch_size, seq_len)
            
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            
            if self.gradient_clipping_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)
            
            self.optimizer.step()
            
            # Step LR scheduler if it's per-step (e.g., CosineAnnealingLR)
            scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
            if self.lr_scheduler and scheduler_config.get('update_frequency', 'epoch') == 'step':
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): # ReduceLROnPlateau steps per epoch
                    self.lr_scheduler.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % (num_batches // 10 if num_batches > 10 else 1) == 0: # Log ~10 times per epoch
                logger.info(f"  Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / num_batches
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Finished Training Epoch {self.current_epoch + 1}. Average Loss: {avg_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
        return avg_epoch_loss

    def _validate_one_epoch(self):
        if not self.valid_loader:
            logger.info("No validation loader provided. Skipping validation.")
            return None, None

        self.model.eval()
        total_val_loss = 0
        num_batches = len(self.valid_loader)
        
        logger.info(f"Starting Validation for Epoch {self.current_epoch + 1}")
        epoch_start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_loader):
                input_ids = batch_data["input_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_val_loss)) # Calculate perplexity from average loss
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Finished Validation for Epoch {self.current_epoch + 1}. Average Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}. Duration: {epoch_duration:.2f}s")
        
        return avg_val_loss, perplexity

    def train(self):
        num_epochs = self.config.get('training', {}).get('num_epochs', 10)
        logger.info(f"Starting training for {num_epochs} epochs.")

        for epoch in range(self.current_epoch, num_epochs): # Resume from self.current_epoch
            self.current_epoch = epoch
            
            train_loss = self._train_one_epoch()
            val_loss, val_perplexity = self._validate_one_epoch()
            
            # Step LR scheduler if it's per-epoch
            scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
            if self.lr_scheduler and scheduler_config.get('update_frequency', 'epoch') == 'epoch':
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss if val_loss is not None else float('inf'))
                else:
                    self.lr_scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"End of Epoch {self.current_epoch + 1}. Current LR: {current_lr:.6e}")

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_perplexity = val_perplexity if val_perplexity is not None else self.best_val_perplexity
                logger.info(f"New best validation loss: {self.best_val_loss:.4f} (Perplexity: {self.best_val_perplexity:.2f}). Saving model...")
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch+1}_best.pt")
            
        logger.info("Training finished.")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}, Best validation perplexity: {self.best_val_perplexity:.2f}")
        return self.best_val_loss, self.best_val_perplexity


    def save_checkpoint(self, file_name: str = "model_checkpoint.pt"):
        checkpoint_dir = Path(self.config.get('training', {}).get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / file_name
        
        save_obj = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'config': self.config 
        }
        if self.lr_scheduler:
            save_obj['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        torch.save(save_obj, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        load_path = Path(checkpoint_path)
        if not load_path.exists():
            logger.error(f"Checkpoint file not found: {load_path}")
            return

        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0) # Default to 0 if not found
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_perplexity = checkpoint.get('best_val_perplexity', float('inf'))
            
            if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            self.model.to(self.device) 
            logger.info(f"Loaded checkpoint from {load_path}. Resuming from epoch {self.current_epoch}.")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {load_path}: {e}", exc_info=True)


if __name__ == '__main__':
    logger.info("Running train_loop.py example...")

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=100, d_model=32): # Removed seq_len as it's dynamic
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc = nn.Linear(d_model, vocab_size) 
            self.vocab_size = vocab_size # Used for dummy output generation

        def forward(self, input_ids, attention_mask=None): # Add attention_mask
            # input_ids shape: (batch, seq_len)
            # attention_mask shape: (batch, seq_len) - not used by this dummy model
            # In a real model, attention_mask would be used by attention layers.
            
            # Dummy logic: just use embeddings and project.
            # This is not a realistic model for sequence processing.
            embeddings = self.embedding(input_ids) # (batch, seq_len, d_model)
            # To make it slightly more "realistic" for loss calculation, let's average embeddings
            # and then project, so output is (batch, vocab_size) if labels are (batch).
            # However, our Trainer expects (batch, seq_len, vocab_size) for logits.
            # So, let's just project each token embedding.
            logits = self.fc(embeddings) # (batch, seq_len, vocab_size)
            return logits

    class DummyDataset(Dataset):
        def __init__(self, num_samples=100, seq_len=64, vocab_size=100):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size
            self.input_ids_data = torch.randint(0, vocab_size, (num_samples, seq_len))
            self.labels_data = torch.randint(0, vocab_size, (num_samples, seq_len)) # Dummy labels
            # For CausalLM, labels are often input_ids shifted, but for this dummy, random is fine.

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            input_ids_chunk = self.input_ids_data[idx]
            labels_chunk = self.labels_data[idx]
            # Create a dummy attention mask (all 1s, meaning all tokens are attended to)
            attention_mask_chunk = torch.ones_like(input_ids_chunk)
            return {"input_ids": input_ids_chunk, 
                    "attention_mask": attention_mask_chunk, 
                    "labels": labels_chunk}

    dummy_config_dict = {
        'model': {'vocab_size': 100, 'd_model': 32}, 
        'data': {'batch_size': 8, 'sequence_length': 64, 'num_workers': 0, 'pin_memory': False},
        'training': {
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'num_epochs': 2,
            'gradient_clipping': 1.0,
            'checkpoint_dir': './dummy_checkpoints_train_loop_final',
            'lr_scheduler': {
                'type': 'cosine_annealing', # Example scheduler
                'min_lr': 1e-5,
                'update_frequency': 'step' # Can be 'step' or 'epoch'
            }
        },
        'benchmarking': {'batch_size_inference': 8}
    }
    
    checkpoint_dir_path = Path(dummy_config_dict['training']['checkpoint_dir'])
    if checkpoint_dir_path.exists():
        shutil.rmtree(checkpoint_dir_path)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        model = DummyModel(
            vocab_size=dummy_config_dict['model']['vocab_size'],
            d_model=dummy_config_dict['model']['d_model']
        )
        
        train_ds = DummyDataset(
            num_samples=128, 
            seq_len=dummy_config_dict['data']['sequence_length'], 
            vocab_size=dummy_config_dict['model']['vocab_size']
        )
        valid_ds = DummyDataset(
            num_samples=64, 
            seq_len=dummy_config_dict['data']['sequence_length'], 
            vocab_size=dummy_config_dict['model']['vocab_size']
        )

        trainer = Trainer(config=dummy_config_dict, model=model, train_dataset=train_ds, valid_dataset=valid_ds)
        
        logger.info("Starting dummy training...")
        trainer.train()
        logger.info("Dummy training finished.")

        trainer.save_checkpoint("dummy_final.pt")
        # Create a new trainer instance to test loading
        model_new = DummyModel(
            vocab_size=dummy_config_dict['model']['vocab_size'],
            d_model=dummy_config_dict['model']['d_model']
        )
        trainer_new = Trainer(config=dummy_config_dict, model=model_new, train_dataset=train_ds, valid_dataset=valid_ds)
        trainer_new.load_checkpoint(checkpoint_dir_path / "dummy_final.pt")
        logger.info("Dummy checkpoint save/load test complete. Loaded epoch: %s", trainer_new.current_epoch)
        # Try one more epoch to ensure scheduler and optimizer states were reloaded correctly
        # trainer_new.train() # This would require num_epochs in config to be > loaded epoch

    except Exception as e:
        logger.error(f"Error in train_loop.py example: {e}", exc_info=True)
    finally:
        if checkpoint_dir_path.exists():
           shutil.rmtree(checkpoint_dir_path)
           logger.info(f"Cleaned up dummy checkpoint directory: {checkpoint_dir_path}")
    
    logger.info("train_loop.py example finished.")
