import os
import argparse
from dotenv import load_dotenv

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

load_dotenv("../.env")
PRETRAINED_PATH = os.getenv('PRETRAINED_PATH')

class WebAttackBertClassifier(pl.LightningModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
        self.tokenizer.model_max_length = 512
        self.model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_PATH, torch_dtype="auto")

        self.args = args
        self.lr = args.lr if args is not None else 0.0001

        self.train_step_outputs = []
        self.validation_step_outputs = []
        
        print(f"Acceleration Device:: {self.device}")

    def forward(self, texts, labels=None):
        """
        Forward pass of the model
        Args:
            - texts : input texts
            - labels : labels of the input texts
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True)
        # print(inputs.keys()) # -> dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        outputs = self.model(**inputs, labels=labels)
        return outputs

    def configure_optimizers(self):
        """
        Configure optimizers
        This method is used to configure the optimizers of the model by using the learning rate
        for specific parameter of the model.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=2, # Period of learning rate decay. (unit: epoch)
            gamma=0.35)  # Multiplicative factor of learning rate decay.

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'f1/val',
            }
        }

    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        Args:
            - batch : batch of the data
            - batch_idx : index of the batch
        """
        texts, labels = batch
        outputs = self(texts, labels=labels)
        # print(outputs.keys()) # -> odict_keys(['loss', 'logits'])

        loss, logits = outputs['loss'], outputs['logits']
        self.train_step_outputs.append(loss)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model, used to compute the metrics
        Args:
            - batch : batch of the data
            - batch_idx : index of the batch
        """
        texts, labels = batch
        outputs = self(texts, labels=labels)
        # print(outputs.keys()) # -> odict_keys(['loss', 'logits'])

        loss, logits = outputs['loss'], outputs['logits']
        output_scores = torch.softmax(logits, dim=-1)
        output = (loss, output_scores, labels)
        
        self.validation_step_outputs.append(output)
        self.log("loss/val", loss)
        return output
    
    def on_train_epoch_end(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

    def on_validation_epoch_end(self):
        """
        End of the validation epoch, this method will be called at the end of the validation epoch,
        it will compute the multiple metrics of classification problem.
        """
        val_preds = torch.tensor([], device=self.device)
        val_scores = torch.tensor([], device=self.device)
        val_labels = torch.tensor([], device=self.device)

        for item in self.validation_step_outputs:
            loss, output_scores, labels = item

            predictions = torch.argmax(output_scores, dim=-1)
            val_preds = torch.cat((val_preds, predictions), dim=0)
            if output_scores.size(1) > 1:
                val_scores = torch.cat((val_scores, output_scores[:, 1]), dim=0)
            else:
                val_scores = torch.cat((val_scores, output_scores[:, 0]), dim=0)
            val_labels = torch.cat((val_labels, labels), dim=0)
        
        val_preds = val_preds.cpu().numpy()
        val_scores = val_scores.cpu().numpy()
        val_labels = val_labels.cpu().numpy()

        try:
            reports = classification_report(val_labels, val_preds, output_dict=True)
            auc = roc_auc_score(val_labels, val_scores)
            accuracy = accuracy_score(val_labels, val_preds)
        except ValueError as e:
            print(f"Error in metrics calculation: {e}")
            auc = 0.0
            accuracy = 0.0
            reports = {"weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}}

        print(classification_report(val_labels, val_preds))
        self.log("auc/val", auc, prog_bar=True)
        self.log("accuracy/val", accuracy, prog_bar=True)
        self.log("precision/val", reports["weighted avg"]["precision"])
        self.log("recall/val", reports["weighted avg"]["recall"])
        self.log("f1/val", reports["weighted avg"]["f1-score"])

        # Clear _step_outputs
        self.train_step_outputs.clear()
        self.validation_step_outputs.clear()

        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WebAttackBertClassifier")
        parser.add_argument('--lr', type=float, default=0.0001)
        return parent_parser
