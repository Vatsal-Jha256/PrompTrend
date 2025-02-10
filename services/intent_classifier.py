from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from typing import List, Tuple
from datasets import Dataset
import os

class IntentClassifier:
    def __init__(
        self,
        model_path: str = "bert-base-uncased",
        num_labels: int = 10,
        use_pretrained: bool = True
    ):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            if use_pretrained and os.path.exists(f"{model_path}/pytorch_model.bin"):
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels
                )
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=num_labels
                )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except ImportError as e:
            if "accelerate" in str(e):
                raise ImportError(
                    "Missing required dependency 'accelerate'. "
                    "Please install it using: pip install accelerate>=0.21.0"
                )
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize classifier: {str(e)}")

    def train(self, training_data: List[dict], epochs: int = 3):
        """Train the model on new data"""
        try:
            # Convert training data to datasets
            dataset = Dataset.from_dict({
                'text': [item.text for item in training_data],
                'label': [item.label for item in training_data]
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
