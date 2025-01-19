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


model_path = "THUDM/glm-4-voice-9b"
tokenizer_path = "THUDM/glm-4-voice-tokenizer"
device = "cuda"

whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

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

import pickle
with open("train_texts.pkl", "wb") as f:
    pickle.dump(train_texts, f)
