# ========================DATA-MODULE===============================

import os
import argparse
import platform
import numpy as np
from typing import Optional
from dotenv import load_dotenv

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler, random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
        # self.data_file = data_file            # Path to the train dataset
        self.train_file = train_file          # Path to the train dataset
        self.val_file = val_file              # Path to the valid dataset
        self.test_file = test_file            # Path to the test dataset
        self.text_column = text_column        # Name of the text column
        self.label_column = label_column      # Name of the label column
        
        self.args = args
        self.batch_size = args.batch_size                      
        self.train_data, self.val_data, self.test_data = None, None, None

    # def setup(self, stage: Optional[str] = None):
    #     """
    #     Prepare the dataset for training and validation.
    #     Args:
    #         stage (str, optional): Stage of the process (e.g., 'fit', 'test').
    #     """
    #     if not os.path.exists(self.data_file):
    #         raise FileNotFoundError(f"Data file not found: {self.data_file}")
    #     full_dataset = WebAttackDataSet(self.data_file, self.text_column, self.label_column)
    #     if len(full_dataset) == 0:
    #         raise ValueError(f"Dataset is empty: {self.data_file}")

    #     # Split the dataset into training and validation sets
    #     train_size = round(len(full_dataset) * self.args.train_ratio)  # defaul 80% for training
    #     val_size = len(full_dataset) - train_size                      # defaul 20% for validation
    #     self.train_data, self.val_data = random_split(full_dataset, [train_size, val_size])
        
    def setup(self, stage: Optional[str] = None):
        """
        Prepare the dataset for training and validation.
        Args:
            stage (str, optional): Stage of the process (e.g., 'fit', 'test').
        """
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Train file not found: {self.train_file}")
        if not os.path.exists(self.val_file):
            raise FileNotFoundError(f"Validation file not found: {self.val_file}")
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test file not found: {self.test_file}")

        # Load datasets
        self.train_data = WebAttackDataSet(self.train_file, self.text_column, self.label_column)
        self.val_data = WebAttackDataSet(self.val_file, self.text_column, self.label_column)
        self.test_data = WebAttackDataSet(self.test_file, self.text_column, self.label_column)

        # Validate datasets are not empty
        if len(self.train_data) == 0:
            raise ValueError(f"Train dataset is empty: {self.train_file}")
        if len(self.val_data) == 0:
            raise ValueError(f"Validation dataset is empty: {self.val_file}")
        if len(self.test_data) == 0:
            raise ValueError(f"Test dataset is empty: {self.test_file}")
        
    def _compute_class_weights(self, dataset):
        """
        Compute weights for each class in the dataset for balanced sampling.
        Args:
            dataset: Dataset to compute class weights for.
        Returns:
            torch.Tensor: Weights for each sample in the dataset.
        """
        print(f"Number: {len(dataset)}")
        labels = [label for _, label in dataset]    # Extract labels from the dataset
        labels = np.array(labels)                   # Convert to numpy array

        # Count the number of samples for each class
        class_sample_count = np.array([np.sum(labels == y) for y in np.unique(labels)])

        # Compute the weight for each class
        class_weights = 1.0 / class_sample_count

        # Compute the weight for each sample
        sample_weights = np.array([class_weights[y] for y in labels])
        return torch.from_numpy(sample_weights).double()

    def train_dataloader(self):
        """
        Returns: DataLoader for training with weighted random sampling.
        """
        sample_weights = self._compute_class_weights(self.train_data)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            self.train_data, 
            sampler=sampler, 
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
        # parser.add_argument('--train_ratio', type=float, default=0.8, help="Train/valid ratio for training")
        parser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataloaders")
        parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for dataloaders")
        return parent_parser
    
def main():
    #######################################################################################################
    parser = argparse.ArgumentParser()

    parser = WebAttackDataModule.add_model_specific_args(parser)
    parser = WebAttackBertClassifier.add_model_specific_args(parser)

    # Add Trainer-specific arguments
    parser.add_argument('--fast-dev-run', type=lambda x: x.lower() == 'true', default=False, 
                        help='Run a quick test (enables fast_dev_run). Pass True or False.')

    args = parser.parse_args()

    #######################################################################################################
    # Load environment variables from the .env file
    load_dotenv("../.env")

    # Access the environment variables
    # DATA_FILE = os.getenv('DATA_CSV_FILE')
    TRAIN_CSV_FILE = os.getenv('TRAIN_CSV_FILE')
    VAL_CSV_FILE = os.getenv('VAL_CSV_FILE')
    TEST_CSV_FILE = os.getenv('TEST_CSV_FILE')
    TEXT_COLUMN = os.getenv('TEXT_COLUMN')
    LABEL_COLUMN = os.getenv('LABEL_COLUMN')
    MODEL_CHECKPOINT_DIR = os.getenv('MODEL_CHECKPOINT_DIR')
    TENSORBOARD_LOGGER_DIR = os.getenv('TENSORBOARD_LOGGER_DIR')

    #######################################################################################################
    model = WebAttackBertClassifier(args=args)
    # dm = WebAttackDataModule(DATA_FILE, TEXT_COLUMN, LABEL_COLUMN, args)
    dm = WebAttackDataModule(TRAIN_CSV_FILE, 
                             VAL_CSV_FILE, 
                             TEST_CSV_FILE, 
                             TEXT_COLUMN, 
                             LABEL_COLUMN, args)
    dm.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINT_DIR,
        filename="wad-bert-tiny-{epoch:02d}-{f1:.2f}",
        save_top_k=3,
        monitor='f1/val',
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=TENSORBOARD_LOGGER_DIR)

    print(f"fast_dev_run={args.fast_dev_run}")
    trainer = pl.Trainer(
        min_epochs=0,                            # Minimum number of epochs to train (0 means no minimum)
        max_epochs=4,                            # Maximum number of epochs to train
        # precision=16,                            # Enable mixed precision training (16-bit)
        logger=tb_logger,                        # Use TensorBoard logger for logging metrics
        val_check_interval=1.0,                  # How often to check the validation set, range [0.0, 1.0] to check.
        callbacks=[
            checkpoint_callback,                 # Save model checkpoints during training
            EarlyStopping('f1/val',              # Stop training if 'f1/val' doesn't improve
            patience=3)],                        # After 3 epochs of no improvement, stop training early
        fast_dev_run=args.fast_dev_run,          # Use --quick to enable fast_dev_run for quick debugging
        accelerator="auto",                      # Automatically select the accelerator (GPU/CPU/MPS)
        devices="auto",                          # Automatically use the available devices (e.g., GPU/CPU)
    )

    trainer.fit(model, dm)
    if checkpoint_callback.best_model_path:
        print('Best checkpoint:', checkpoint_callback.best_model_path)
    else:
        print('No checkpoint was saved.')

    print(f"Run TensorBoard with: tensorboard --logdir={TENSORBOARD_LOGGER_DIR}")
    return model

if __name__ == '__main__':
    main()