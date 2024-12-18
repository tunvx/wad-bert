import os
import argparse
from dotenv import load_dotenv

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

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
        self.test_step_outputs = []

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
        # AdamW optimizer for the model's parameters
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # StepLR scheduler: reduces learning rate by gamma every 'step_size' epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.35)
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                'scheduler': scheduler,
                'monitor': 'f1_val',
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
        self.log("loss_train", loss)
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
        self.log("loss_val", loss)
        return output
    
    def test_step(self, batch, batch_idx):
        """
        Test step of the model, used to compute metrics during the testing phase.
        Args:
            - batch : batch of the data
            - batch_idx : index of the batch
        """
        texts, labels = batch
        outputs = self(texts, labels=labels)

        loss, logits = outputs['loss'], outputs['logits']
        output_scores = torch.softmax(logits, dim=-1)
        output = (loss, output_scores, labels)
        
        self.test_step_outputs.append(output)
        self.log("loss_test", loss)
        return output
    
    def on_fit_start(self):
        print(f"Acceleration Device:: {self.device}")

    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        print(f"Current Learning Rate: {lr}")

    def on_train_epoch_end(self):
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        """
        End of the validation epoch, this method will be called at the end of the validation epoch,
        it will compute the multiple metrics of classification problem.
        """
        preds, scores, labels = self._process_epoch_end(self.validation_step_outputs, self.device)
        reports, auc, accuracy = self._compute_metrics(preds, scores, labels)
      
        self.log("accuracy_val", accuracy)
        self.log("precision_val", reports["weighted avg"]["precision"])
        self.log("recall_val", reports["weighted avg"]["recall"])
        self.log("f1_val", reports["weighted avg"]["f1-score"])
        self.log("auc_val", auc)
        print(classification_report(labels, preds))
        
        # Clear _step_outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """
        End of the test epoch, this method will be called at the end of the test epoch,
        it will compute the multiple metrics of classification problem.
        """
        preds, scores, labels = self._process_epoch_end(self.test_step_outputs, self.device)
        reports, auc, accuracy = self._compute_metrics(preds, scores, labels)
        
        self.log("accuracy_test", accuracy)
        self.log("precision_test", reports["weighted avg"]["precision"])
        self.log("recall_test", reports["weighted avg"]["recall"])
        self.log("f1_test", reports["weighted avg"]["f1-score"])
        self.log("auc_test", auc)
        print(classification_report(labels, preds))

        # Clear _step_outputs after test
        self.test_step_outputs.clear()
    
    @staticmethod  
    def _process_epoch_end(step_outputs, device):
        preds = torch.tensor([], device=device)
        scores = torch.tensor([], device=device)
        labels = torch.tensor([], device=device)

        for item in step_outputs:
            loss, output_scores, label = item
            
            predictions = torch.argmax(output_scores, dim=-1)
            preds = torch.cat((preds, predictions), dim=0)
            scores = torch.cat((scores, output_scores[:, -1]), dim=0)   # Use the last column for scores
            labels = torch.cat((labels, label), dim=0)
        
        return preds.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
    
    @staticmethod
    def _compute_metrics(preds, scores, labels):
        try:
            reports = classification_report(labels, preds, output_dict=True)
            auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, preds)
        except ValueError as e:
            print(f"Error in metrics calculation: {e}")
            reports = {"weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}}
            auc = accuracy = 0.0
        
        return reports, auc, accuracy
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WebAttackBertClassifier")
        parser.add_argument('--lr', type=float, default=0.0001)
        return parent_parser
