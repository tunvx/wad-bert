# ========================DATA-MODULE===============================

import os
import argparse
import platform
import numpy as np
from typing import Optional
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from dataset import WebAttackDataSet
from model import WebAttackBertClassifier

# Disable parallelism in the Hugging Face tokenizers library to prevent potential conflicts or deadlocks 
# when using multiprocessing or multi-threading, particularly in environments where processes are forked. 
# This setting only affects the tokenizers' internal parallelism and does not impact GPU-based parallelism 
# or other parts of the program.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WebAttackDataModule(pl.LightningDataModule):
    """
    Module class for web acttack analysis. this class is used to load the data to the model. 
    It is a subclass of LightningDataModule. 
    """

    def __init__(self, train_file: str, val_file: str, test_file: str, text_column: str, label_column: str, args: argparse.Namespace):
        """
        Initialize the DataModule.
        Args:
            data_file (str): Path to the CSV file containing the dataset.
            text_column (str): Name of the column containing text data.
            label_column (str): Name of the column containing labels.
            args (argparse.Namespace): Command-line arguments containing batch size, train ratio, etc.
        """
        super().__init__()
        self.train_file = train_file          # Path to the train dataset
        self.val_file = val_file              # Path to the valid dataset
        self.test_file = test_file            # Path to the test dataset
        self.text_column = text_column        # Name of the text column
        self.label_column = label_column      # Name of the label column
        
        self.args = args
        self.verbose = args.verbose             
        self.batch_size = args.batch_size         
        self.train_data, self.val_data, self.test_data = None, None, None
    
    def setup(self, stage: Optional[str] = None):
        """
        Load the pre-split datasets for the appropriate stage.
        Args:
            stage (str, optional): Stage of the process (e.g., 'fit', 'test').
        """
        # Load datasets for training/validation
        if stage == "fit" or stage is None:
            self.train_data = WebAttackDataSet(self.train_file, self.text_column, self.label_column)
            self.val_data = WebAttackDataSet(self.val_file, self.text_column, self.label_column)
            if self.verbose:
                print(f"\nTrain dataset: {self.train_file}\n Number of samples: {len(self.train_data)}")
                print(f"\nValidation dataset: {self.val_file}\n Number of samples: {len(self.val_data)}")

        # Load dataset for testing
        if stage == "test" or stage is None:
            self.test_data = WebAttackDataSet(self.test_file, self.text_column, self.label_column)
            if self.verbose:
                print(f"\nTest dataset: {self.test_file}\n Number of samples: {len(self.test_data)}")
        
    def _compute_class_weights(self, dataset, verbose=False):
        """
        Compute class weights for imbalanced datasets.
        Args:
            dataset: Dataset to compute class weights for.
        Returns:
            torch.Tensor: Weights for each sample in the dataset.
        """
        labels = np.array([label for _, label in dataset])
        _, class_counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        if self.verbose:
            print(f'Class Counts: {class_counts}')
            print(f'Class Weights: {class_weights}')
        return torch.from_numpy(sample_weights).double()

    def train_dataloader(self):
        """
        Returns: DataLoader for training with weighted random sampling.
        """
        # sample_weights = self._compute_class_weights(self.train_data, verbose=True)
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            self.train_data, 
            sampler=None,           # None because wad dataset is balanced
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=self.args.num_workers, 
            persistent_workers=True # Keep the DataLoader workers alive between epochs
        )

    def val_dataloader(self):
        """
        Returns: DataLoader for validation.
        """
        return DataLoader(
            self.val_data, 
            shuffle=False, 
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=self.args.num_workers, 
            persistent_workers=True  # Keep the DataLoader workers alive between epochs
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            shuffle=False, 
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=self.args.num_workers, 
            persistent_workers=True  # Keep the DataLoader workers alive between epochs
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add command-line arguments specific to the DataModule.
        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.
        Returns:
            argparse.ArgumentParser: Updated argument parser.
        """
        parser = parent_parser.add_argument_group("WebAttackDataModule")
        parser.add_argument('--batch-size', type=int, default=32, help="Batch size for dataloaders")
        parser.add_argument('--num-workers', type=int, default=2, help="Number of workers for dataloaders")
        return parent_parser
    
def main():
    parser = argparse.ArgumentParser()

    parser = WebAttackDataModule.add_model_specific_args(parser)
    parser = WebAttackBertClassifier.add_model_specific_args(parser)

    # Add Trainer-specific arguments
    parser.add_argument("--max-epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--verbose', action='store_true', help="Print debugging information") 
    parser.add_argument('--fast-dev-run', action='store_true', 
                        help='Run a quick test (enables fast_dev_run). Pass True or False.')

    args = parser.parse_args()

    load_dotenv("../.env", override=True)

    # DATA_FILE = os.getenv('DATA_CSV_FILE')
    TRAIN_CSV_FILE = os.getenv('TRAIN_CSV_FILE')
    VAL_CSV_FILE = os.getenv('VAL_CSV_FILE')
    TEST_CSV_FILE = os.getenv('TEST_CSV_FILE')
    TEXT_COLUMN = os.getenv('TEXT_COLUMN')
    LABEL_COLUMN = os.getenv('LABEL_COLUMN')
    MODEL_CHECKPOINT_DIR = os.getenv('MODEL_CHECKPOINT_DIR')
    TENSORBOARD_LOGGER_DIR = os.getenv('TENSORBOARD_LOGGER_DIR')

    ############################################################################
    model = WebAttackBertClassifier(args=args)
    # dm = WebAttackDataModule(DATA_FILE, TEXT_COLUMN, LABEL_COLUMN, args)
    dm = WebAttackDataModule(TRAIN_CSV_FILE, 
                             VAL_CSV_FILE, 
                             TEST_CSV_FILE, 
                             TEXT_COLUMN, 
                             LABEL_COLUMN, args)
    # dm.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINT_DIR,
        filename="wad-bert-tiny-{epoch:02d}-{f1_val:.2f}",
        save_top_k=3,
        monitor='f1_val',
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=TENSORBOARD_LOGGER_DIR)

    print(f"fast_dev_run={args.fast_dev_run}")
    trainer = pl.Trainer(
        min_epochs=0,                       # Minimum number of epochs to train (0 means no minimum)
        max_epochs=args.max_epochs,         # Maximum number of epochs to train
        # precision=16,                     # Enable mixed precision training (16-bit)
        logger=tb_logger,                   # Use TensorBoard logger for logging metrics
        val_check_interval=1.0,             # How often to check the validation set, range [0.0, 1.0] to check.
        callbacks=[
            checkpoint_callback,            # Save model checkpoints during training
            lr_monitor_callback,
            EarlyStopping('f1_val',         # Stop training if 'f1/val' doesn't improve
            patience=3)],                   # After 3 epochs of no improvement, stop training early
        fast_dev_run=args.fast_dev_run,     # Use --quick to enable fast_dev_run for quick debugging
        accelerator="auto",                 # Automatically select the accelerator (GPU/CPU/MPS)
        devices="auto",                     # Automatically use the available devices (e.g., GPU/CPU)
    )

    print("\nRunning training...")
    trainer.fit(model, dm)
    if checkpoint_callback.best_model_path:
        print('Best checkpoint:', checkpoint_callback.best_model_path)
    else:
        print('No checkpoint was saved.')
    
    print("\nRunning testing...")
    trainer.test(model, dm)

    print(f"Run TensorBoard with: tensorboard --logdir={TENSORBOARD_LOGGER_DIR}")
    return model

if __name__ == '__main__':
    main()