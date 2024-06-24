import os
import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import zipfile


class BertClassificator:
    def __init__(self, data_path='data/train.json.zip', model_name='bert-base-multilingual-cased', num_labels=3, model_dir='model'):
        self.data_path = data_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)

        # Load model if available, else train
        if not self.load_model():
            self.data = self.load_data()
            self.train_dataloader, self.val_dataloader = self.prepare_datasets()
            self.train()

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            print("Model and tokenizer loaded.")
            return True
        return False

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer')
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print("Model and tokenizer saved.")

    def load_data(self):
        # try:
        #     with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
        #         zip_ref.extractall('data')
        # except Exception as e:
        #     print(f"Error extracting zip file: {e}")
        #     return []

        data = []
        try:
            with open('data/qas/combined_dataset_with_responses_and_classification.json', 'r') as file:
                for line in file:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return []

        return pd.DataFrame(data)

    def preprocess_data(self, text_list, max_length=128):
        input_ids, attention_masks = [], []

        for text in text_list:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    def prepare_datasets(self):
        input_ids, attention_masks = self.preprocess_data(self.data['ModelResponse'].to_list())
        labels = self.data['Classification'].apply(lambda x: 0 if x == 'neither' else 1 if x == 'yes' else 2).values
        labels = torch.tensor(labels).to(self.device)

        train_size = int(0.8 * len(self.data))
        val_size = len(self.data) - train_size

        train_dataset, val_dataset = random_split(TensorDataset(input_ids, attention_masks, labels), [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        return train_dataloader, val_dataloader

    def train(self, epochs=4, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_dataloader:
                b_input_ids, b_input_mask, b_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.2f}")

            self.evaluate()

        # Save the model after training
        self.save_model()

    def evaluate(self):
        self.model.eval()
        val_accuracy, val_loss = 0, 0

        for batch in self.val_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy += accuracy

        avg_val_accuracy = val_accuracy / len(self.val_dataloader)
        avg_val_loss = val_loss / len(self.val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {avg_val_accuracy:.2f}%')

    def predict(self, text):
        self.model.eval()
        inputs = self.preprocess_data([text])
        input_ids, attention_masks = inputs[0].to(self.device), inputs[1].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        return prediction