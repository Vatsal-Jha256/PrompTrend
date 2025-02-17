from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from typing import List, Tuple
from datasets import Dataset
import os
from transformers import BertConfig  # Add this import
class IntentClassifier:
    def __init__(self, model_path="bert-base-uncased", num_labels=10, use_pretrained=True):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
            
            if use_pretrained and os.path.exists(f"{model_path}/pytorch_model.bin"):
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path,
                    config=config
                )
            else:
                self.model = BertForSequenceClassification(config)
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                self.model.to(self.device)
            except RuntimeError as e:
                print(f"Could not use CUDA: {str(e)}, falling back to CPU")
                self.device = torch.device('cpu')
                self.model.to(self.device)
        except Exception as e:
            raise Exception(f"Error initializing BERT model: {str(e)}")
    def train(self, training_data: List[dict], epochs: int = 3):
        """Train the model on new data"""
        try:
            if not training_data:
                raise ValueError("Training data cannot be empty")
            # Access fields using dot notation instead of dictionary access
            labels = [item['label'] for item in training_data]
            if any(label < 0 or label >= self.model.config.num_labels for label in labels):
                raise ValueError("Labels must be between 0 and num_labels-1")
            
            dataset = Dataset.from_dict({
                'text': [item['text'] for item in training_data],  # Change from item.text
                'label': labels
            })

            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Check for accelerate before creating trainer
            try:
                import accelerate
            except ImportError:
                raise ImportError(
                    "The 'accelerate' package is required for training. "
                    "Please install it using: pip install accelerate>=0.21.0"
                )

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                save_steps=1000,
                save_total_limit=2,
                logging_steps=100,
                logging_dir="./logs",
                logging_first_step=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            
            trainer.train()
            
            # Save the model
            os.makedirs("./trained_model", exist_ok=True)
            self.model.save_pretrained("./trained_model")
            self.tokenizer.save_pretrained("./trained_model")
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def classify(self, text: str) -> Tuple[int, float]:
        """Classify text into intent category and return confidence score"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probs).item()
                confidence = probs[0][predicted_class].item()
                
            return predicted_class, confidence
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")
