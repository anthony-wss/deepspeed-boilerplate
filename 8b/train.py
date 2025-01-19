# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokenized = tokenizer(self.dataset[idx]['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        }

class CustomDataCollator:
    """Custom data collator for batching audio and text inputs."""

    def __init__(self):
        pass

    def __call__(self, batch):
        return {
            "input_ids": torch.cat([item["input_ids"] for item in batch]),
            "attention_mask": torch.cat([item["attention_mask"] for item in batch]),
            "labels": torch.cat([item["labels"] for item in batch])
        }

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print("Model loaded")

tokenizer.pad_token = tokenizer.eos_token

train_ds = load_dataset("timdettmers/openassistant-guanaco", split='train[:10%]')
test_ds = load_dataset("timdettmers/openassistant-guanaco", split='test[:10%]')

print("Dataset loaded")

tokenized_train_ds = CustomDataset(train_ds)
tokenized_test_ds = CustomDataset(test_ds)

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_strategy="steps",
    logging_steps=1,
    max_steps = 50,
    fp16=True,
    gradient_checkpointing=True,
    deepspeed="ds_config.json",
    label_names=["labels"]
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    data_collator=CustomDataCollator()
)
trainer.train()
