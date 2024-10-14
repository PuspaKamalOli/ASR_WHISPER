import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# Set environment variable to allow CPU fallback for unsupported MPS operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class WhisperASRTrainer:
    def __init__(self, model_name: str, language: str, task: str):
        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.feature_extractor, self.processor = self._initialize_processor()
        self.metric = evaluate.load("wer")

    def _initialize_processor(self):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
        processor = WhisperProcessor.from_pretrained(self.model_name, 
                                                     language='en', task=self.task)
        return feature_extractor, processor

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, 
                                               Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def load_preprocessed_data(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    def train_model(self, voice_data, output_dir,
                    model_save_path, pickle_save_path):
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        model.config.language = self.language
        model.config.task = self.task
        model.config.forced_decoder_ids = None

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=10,  #10000
            gradient_checkpointing=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=50,#1000
            eval_steps=50,#1000
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        # Explicitly set use_cache to False if gradient_checkpointing is enabled
        if training_args.gradient_checkpointing:
            model.config.use_cache = False

        data_collator = self.DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )

        def move_to_device(batch, device):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            return batch

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=voice_data["train"],
            eval_dataset=voice_data["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.tokenizer,
        )

        def prediction_step(model, inputs, prediction_loss_only, ignore_keys):
            inputs = move_to_device(inputs, self.device)
            return trainer.prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        trainer.prediction_step = prediction_step
        trainer.train()

        model.save_pretrained(model_save_path)
        self.processor.save_pretrained(model_save_path)

        with open(pickle_save_path, 'wb') as f:
            pickle.dump(model_save_path, f)

        print(f"Model saved to {pickle_save_path}")

if __name__ == "__main__":
    # Configuration
    model_name = "openai/whisper-tiny"
    language = "English"
    task = "transcribe"
    data_pickle_file = "data.pkl"
    output_dir = "./whisper_ASR"
    model_save_path = "./whisper_ASR_model"
    pickle_save_path = "./whisper_ASR_model.pkl"

    # Initialize Trainer
    trainer = WhisperASRTrainer(model_name=model_name, 
                                language=language, task=task)

    # Load preprocessed dataset
    voice_data = trainer.load_preprocessed_data(data_pickle_file)

    # Train and save model
    trainer.train_model(voice_data, output_dir, model_save_path, 
                        pickle_save_path)
