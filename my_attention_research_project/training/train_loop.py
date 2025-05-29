# train_loop.py
# Contains the main training loop and utilities.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time # Ensure time is imported
import yaml # For config type hinting if used, or general dict
from pathlib import Path
import logging # For logging
import shutil # For __main__ example cleanup

# For type hinting if needed, or __main__ example.
try:
    from ..models.transformer import TransformerEncoder 
except ImportError:
    if __name__ != '__main__': 
        raise
    print("Warning: Could not import TransformerEncoder. This is fine if running the __main__ example with DummyModel.")
    TransformerEncoder = None 

logger = logging.getLogger(__name__) # Use module-level logger
# BasicConfig should ideally be called once at application entry point.
# If this script is run standalone, it's fine. If imported, it might conflict or have no effect if already configured.
# For this project, assuming it's called early enough or this is the main config point for logging.
if not logger.hasHandlers(): # Avoid adding multiple handlers if imported multiple times or run after setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self, config: dict, model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset = None):
        logger.info("Initializing Trainer...")
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

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
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        batch_size = self.config.get('data', {}).get('batch_size', 32)
        num_workers = self.config.get('data', {}).get('num_workers', 0)
        pin_memory = self.config.get('data', {}).get('pin_memory', True if self.device.type == 'cuda' else False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=pin_memory)
        
        if self.valid_dataset:
            eval_batch_size = self.config.get('benchmarking', {}).get('batch_size_inference', batch_size)
            self.valid_loader = DataLoader(
                self.valid_dataset, batch_size=eval_batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=pin_memory)
        else:
            self.valid_loader = None
            
        logger.info(f"Train DataLoader: batch_size={batch_size}, num_examples={len(self.train_dataset)}, num_workers={num_workers}")
        if self.valid_loader:
             logger.info(f"Valid DataLoader: batch_size={eval_batch_size}, num_examples={len(self.valid_dataset)}, num_workers={num_workers}")

        self.lr_scheduler = None
        scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
        if scheduler_config and scheduler_config.get('type'):
            scheduler_type = scheduler_config.get('type').lower()
            if scheduler_type == 'cosine_annealing':
                num_epochs = self.config.get('training', {}).get('num_epochs', 1)
                total_steps = num_epochs * len(self.train_loader)
                eta_min = float(scheduler_config.get('min_lr', 0.0))
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_steps, eta_min=eta_min)
                logger.info(f"Using CosineAnnealingLR scheduler. T_max={total_steps}, eta_min={eta_min}")
            elif scheduler_type == 'reduce_lr_on_plateau':
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode=scheduler_config.get('mode', 'min'),
                    factor=float(scheduler_config.get('factor', 0.1)),
                    patience=int(scheduler_config.get('patience', 10)))
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

    def log_memory_usage(self, context_message: str = ""):
        """Logs current and peak GPU memory usage if CUDA is available."""
        peak_gpu_memory_mb = 0
        if self.device.type == 'cuda':
            # current_memory_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            # logger.info(f"Current GPU Memory usage {context_message}: {current_memory_mb:.2f} MB")
            logger.info(f"Peak GPU Memory Allocated {context_message}: {peak_memory_mb:.2f} MB")
            # Reset peak stats for the next measurement period if desired (e.g., per epoch)
            # torch.cuda.reset_peak_memory_stats(self.device) # Be careful with this
        else:
            logger.info(f"Memory usage logging: CUDA not available. {context_message}")
        return peak_gpu_memory_mb


    def _train_one_epoch(self) -> dict:
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        logger.info(f"Starting Training Epoch {self.current_epoch + 1}")
        epoch_start_time = time.time()
        total_tokens_in_epoch = 0

        for batch_idx, batch_data in enumerate(self.train_loader):
            input_ids = batch_data["input_ids"].to(self.device)
            attention_mask = batch_data["attention_mask"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            
            total_tokens_in_epoch += input_ids.numel() # input_ids.size(0) * input_ids.size(1)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            
            if self.gradient_clipping_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)
            
            self.optimizer.step()
            
            scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
            if self.lr_scheduler and scheduler_config.get('update_frequency', 'epoch') == 'step':
                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % (num_batches // 10 if num_batches > 10 else 1) == 0:
                logger.info(f"  Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / num_batches
        epoch_duration = time.time() - epoch_start_time
        training_tokens_per_sec = total_tokens_in_epoch / epoch_duration if epoch_duration > 0 else 0
        
        logger.info(f"Finished Training Epoch {self.current_epoch + 1}. Average Loss: {avg_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
        logger.info(f"Training throughput: {training_tokens_per_sec:.2f} tokens/sec")
        
        epoch_metrics = {
            'epoch_train_loss': avg_epoch_loss,
            'training_tokens_per_sec': training_tokens_per_sec,
            'epoch_lr': self.optimizer.param_groups[0]['lr'] # Get current LR
        }
        return epoch_metrics

    def _validate_one_epoch(self) -> dict:
        if not self.valid_loader:
            logger.info("No validation loader provided. Skipping validation.")
            return {'perplexity': None, 'eval_loss': None, 'inference_tokens_per_sec': None}

        self.model.eval()
        total_val_loss = 0
        num_batches = len(self.valid_loader)
        
        logger.info(f"Starting Validation for Epoch {self.current_epoch + 1}")
        eval_start_time = time.time()
        total_tokens_in_eval = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_loader):
                input_ids = batch_data["input_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                
                total_tokens_in_eval += input_ids.numel()
                
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item() 
        
        eval_duration = time.time() - eval_start_time
        inference_tokens_per_sec = total_tokens_in_eval / eval_duration if eval_duration > 0 else 0
        
        logger.info(f"Finished Validation for Epoch {self.current_epoch + 1}. Average Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}. Duration: {eval_duration:.2f}s")
        logger.info(f"Inference throughput: {inference_tokens_per_sec:.2f} tokens/sec")
        
        eval_metrics = {
            'perplexity': perplexity,
            'eval_loss': avg_val_loss,
            'inference_tokens_per_sec': inference_tokens_per_sec
        }
        return eval_metrics

    def train(self):
        num_epochs = self.config.get('training', {}).get('num_epochs', 10)
        logger.info(f"Starting training for {num_epochs} epochs.")
        
        all_epoch_summaries = [] # To store metrics from all epochs

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Reset peak memory stats at the beginning of each epoch measurement cycle
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            
            train_metrics = self._train_one_epoch()
            peak_gpu_memory_after_train_mb = self.log_memory_usage(f"after Training Epoch {self.current_epoch + 1}")
            
            eval_metrics = self._validate_one_epoch()
            peak_gpu_memory_after_eval_mb = self.log_memory_usage(f"after Validation Epoch {self.current_epoch + 1}")
            
            # Use the higher of the two peaks for the epoch summary
            epoch_peak_gpu_memory_mb = max(peak_gpu_memory_after_train_mb, peak_gpu_memory_after_eval_mb)

            scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
            if self.lr_scheduler and scheduler_config.get('update_frequency', 'epoch') == 'epoch':
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(eval_metrics['eval_loss'] if eval_metrics['eval_loss'] is not None else float('inf'))
                else:
                    self.lr_scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr'] # Already in train_metrics, but good to have for summary
            
            epoch_summary_metrics = {
                'epoch': self.current_epoch + 1,
                'learning_rate': current_lr,
                'training_loss': train_metrics['epoch_train_loss'],
                'validation_loss': eval_metrics['eval_loss'],
                'perplexity': eval_metrics['perplexity'],
                'training_tokens_per_sec': train_metrics['training_tokens_per_sec'],
                'inference_tokens_per_sec': eval_metrics['inference_tokens_per_sec'],
                'peak_gpu_memory_mb': epoch_peak_gpu_memory_mb
            }
            all_epoch_summaries.append(epoch_summary_metrics) # Store for later use

            logger.info(f"Epoch {epoch_summary_metrics['epoch']} Summary: LR={epoch_summary_metrics['learning_rate']:.6e}, "
                        f"Train Loss={epoch_summary_metrics['training_loss']:.4f}, Valid Loss={epoch_summary_metrics['validation_loss']:.4f}, "
                        f"Perplexity={epoch_summary_metrics['perplexity']:.2f}, Train Tok/s={epoch_summary_metrics['training_tokens_per_sec']:.0f}, "
                        f"Eval Tok/s={epoch_summary_metrics['inference_tokens_per_sec']:.0f}, Peak Mem={epoch_summary_metrics['peak_gpu_memory_mb']:.2f}MB")


            if eval_metrics['eval_loss'] is not None and eval_metrics['eval_loss'] < self.best_val_loss:
                self.best_val_loss = eval_metrics['eval_loss']
                self.best_val_perplexity = eval_metrics['perplexity'] if eval_metrics['perplexity'] is not None else self.best_val_perplexity
                logger.info(f"New best validation loss: {self.best_val_loss:.4f} (Perplexity: {self.best_val_perplexity:.2f}). Saving model...")
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch+1}_best.pt")
            
        logger.info("Training finished.")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}, Best validation perplexity: {self.best_val_perplexity:.2f}")
        
        # This list of dictionaries can be returned or used by a higher-level runner
        return all_epoch_summaries


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
            self.current_epoch = checkpoint.get('epoch', 0) 
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_perplexity = checkpoint.get('best_val_perplexity', float('inf'))
            
            if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            self.model.to(self.device) 
            logger.info(f"Loaded checkpoint from {load_path}. Resuming from epoch {self.current_epoch}.")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {load_path}: {e}", exc_info=True)


if __name__ == '__main__':
    # Ensure logger is configured for __main__
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__) # Use the module's logger

    main_logger.info("Running train_loop.py example...")

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=100, d_model=32): 
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc = nn.Linear(d_model, vocab_size) 
            self.vocab_size = vocab_size 

        def forward(self, input_ids, attention_mask=None): 
            embeddings = self.embedding(input_ids) 
            logits = self.fc(embeddings) 
            return logits

    class DummyDataset(Dataset):
        def __init__(self, num_samples=100, seq_len=64, vocab_size=100):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size
            self.input_ids_data = torch.randint(0, vocab_size, (num_samples, seq_len))
            self.labels_data = torch.randint(0, vocab_size, (num_samples, seq_len)) 
            self.attention_mask_data = torch.ones_like(self.input_ids_data)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids_data[idx], 
                    "attention_mask": self.attention_mask_data[idx], 
                    "labels": self.labels_data[idx]}

    dummy_config_dict = {
        'model': {'vocab_size': 100, 'd_model': 32}, 
        'data': {'batch_size': 8, 'sequence_length': 64, 'num_workers': 0, 'pin_memory': False},
        'training': {
            'optimizer': 'AdamW', 'learning_rate': 1e-3, 'weight_decay': 0.01,
            'num_epochs': 2, 'gradient_clipping': 1.0,
            'checkpoint_dir': './dummy_checkpoints_train_loop_enhanced',
            'lr_scheduler': {'type': 'cosine_annealing', 'min_lr': 1e-5, 'update_frequency': 'step'}
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
        train_ds = DummyDataset(num_samples=128, seq_len=dummy_config_dict['data']['sequence_length'], vocab_size=dummy_config_dict['model']['vocab_size'])
        valid_ds = DummyDataset(num_samples=64, seq_len=dummy_config_dict['data']['sequence_length'], vocab_size=dummy_config_dict['model']['vocab_size'])

        trainer = Trainer(config=dummy_config_dict, model=model, train_dataset=train_ds, valid_dataset=valid_ds)
        
        main_logger.info("Starting dummy training with enhanced logging...")
        epoch_metrics_history = trainer.train() # Capture the history
        main_logger.info("Dummy training finished.")
        
        if epoch_metrics_history:
            main_logger.info("Metrics history collected:")
            for epoch_summary in epoch_metrics_history:
                main_logger.info(f"  Epoch {epoch_summary['epoch']}: {epoch_summary}")
        else:
            main_logger.warning("No metrics history was collected.")


        trainer.save_checkpoint("dummy_final_enhanced.pt")
        model_new = DummyModel(vocab_size=dummy_config_dict['model']['vocab_size'], d_model=dummy_config_dict['model']['d_model'])
        trainer_new = Trainer(config=dummy_config_dict, model=model_new, train_dataset=train_ds, valid_dataset=valid_ds)
        trainer_new.load_checkpoint(checkpoint_dir_path / "dummy_final_enhanced.pt")
        main_logger.info("Dummy checkpoint save/load test complete. Loaded epoch: %s", trainer_new.current_epoch)

    except Exception as e:
        main_logger.error(f"Error in train_loop.py example: {e}", exc_info=True)
    finally:
        if checkpoint_dir_path.exists():
           shutil.rmtree(checkpoint_dir_path)
           main_logger.info(f"Cleaned up dummy checkpoint directory: {checkpoint_dir_path}")
    
    main_logger.info("train_loop.py example finished.")
