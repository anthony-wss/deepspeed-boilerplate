# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset

from speech_tokenizer.utils import extract_speech_token
from src.model import FineTunedGLM4Voice, GLM4VoiceTokenizer, prepare_model_for_lora
from src.dataset import TextDataset
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor, AutoTokenizer
import os
from transformers import Trainer, TrainingArguments

os.environ["WANDB_PROJECT"] = "glm-4-voice-finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

model_path = "THUDM/glm-4-voice-9b"
tokenizer_path = "THUDM/glm-4-voice-tokenizer"
device = "cuda"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.dataset[idx]['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
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

whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

glm_model = FineTunedGLM4Voice(model_path).model
glm_tokenizer = GLM4VoiceTokenizer(model_path).tokenizer
print("Model loaded")

# Prepare `train_texts` consisting of 100 identical audio inputs for debugging
audio_path = "./input-sample-0.wav"
audio_tokens = extract_speech_token(
    whisper_model, feature_extractor, [audio_path]
)[0]  # 12.5 TPS
audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
user_input = audio_tokens
system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>streaming_transcription\n"
train_texts = [prompt] * 100

train_dataset = CustomDataset(train_texts, glm_tokenizer)
print("Dataset loaded")

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
    model=glm_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=CustomDataCollator()
)
trainer.train()
