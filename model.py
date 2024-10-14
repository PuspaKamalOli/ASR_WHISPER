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
    """
    Trainer class for Automatic Speech Recognition (ASR) using the Whisper model.
    
    Handles loading the model, training on preprocessed speech data, and saving the model.
    """
    def __init__(self, model_name: str, language: str, task: str):
        """
        Initialize the WhisperASRTrainer.

        Args:
            model_name (str): The name of the pre-trained Whisper model.
            language (str): The language for which the ASR model will be trained.
            task (str): The specific task to be performed, e.g., 'transcribe'.
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        # Set device to MPS (Metal Performance Shaders) if available, otherwise fall back to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Initialize processor and feature extractor
        self.feature_extractor, self.processor = self._initialize_processor()
        # Load evaluation metric for Word Error Rate (WER)
        self.metric = evaluate.load("wer")

    def _initialize_processor(self):
        """
        Load the feature extractor and processor for the Whisper model.

        Returns:
            feature_extractor: WhisperFeatureExtractor object for audio preprocessing.
            processor: WhisperProcessor object for text and feature processing.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
        processor = WhisperProcessor.from_pretrained(self.model_name, language='en', task=self.task)
        return feature_extractor, processor

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        A data collator that dynamically pads the inputs and labels during training.
        
        This ensures that all inputs in a batch have the same length.
        """
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            """
            Pads the input features and labels for a batch of training examples.

            Args:
                features (List[Dict]): List of input feature dictionaries.

            Returns:
                Dict[str, torch.Tensor]: A batch of padded input features and labels.
            """
            # Pad input features for the model
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Pad labels and mask padded tokens with -100 to ignore them during loss computation
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Ensure that labels start after the decoder start token (for sequence generation tasks)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    def compute_metrics(self, pred):
        """
        Compute Word Error Rate (WER) metric on model predictions.

        Args:
            pred: Predictions from the model.

        Returns:
            Dict[str, float]: A dictionary containing the WER score.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels into strings
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute Word Error Rate (WER)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def load_preprocessed_data(self, pickle_file: str):
        """
        Load preprocessed voice data from a pickle file.

        Args:
            pickle_file (str): Path to the pickle file containing preprocessed data.

        Returns:
            Loaded voice data from the pickle file.
        """
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    def train_model(self, voice_data, output_dir: str, model_save_path: str, pickle_save_path: str):
        """
        Train the Whisper ASR model on the preprocessed dataset and save the trained model.

        Args:
            voice_data: The preprocessed voice dataset to use for training.
            output_dir (str): The directory where training outputs will be saved.
            model_save_path (str): The path to save the trained model.
            pickle_save_path (str): The path to save the model directory in a pickle file.
        """
        # Load Whisper model and configure language and task
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        model.config.language = self.language
        model.config.task = self.task
        model.config.forced_decoder_ids = None

        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=10,  # Number of training steps
            gradient_checkpointing=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=50,  # Save model every 50 steps
            eval_steps=50,  # Evaluate every 50 steps
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        # If gradient checkpointing is enabled, set use_cache to False
        if training_args.gradient_checkpointing:
            model.config.use_cache = False

        # Create a data collator for padding the inputs and labels
        data_collator = self.DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )

        def move_to_device(batch, device):
            """
            Move batch tensors to the specified device (CPU or MPS).

            Args:
                batch (dict): Dictionary containing batch tensors.
                device: The device to move the tensors to.

            Returns:
                dict: The batch with tensors moved to the specified device.
            """
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            return batch

        # Initialize Seq2SeqTrainer with the model, training data, and arguments
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=voice_data["train"],
            eval_dataset=voice_data["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.tokenizer,
        )

        # Modify trainer's prediction step to ensure the inputs are moved to the correct device
        def prediction_step(model, inputs, prediction_loss_only, ignore_keys):
            inputs = move_to_device(inputs, self.device)
            return trainer.prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        trainer.prediction_step = prediction_step

        # Train the model
        trainer.train()

        # Save the trained model and processor
        model.save_pretrained(model_save_path)
        self.processor.save_pretrained(model_save_path)

        # Save the model directory path in a pickle file
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
