import os
import pickle
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def load_process_and_save_dataset(data_path: str, filename: str, 
                                  model_name: str, language: str, task: str) -> None:
    """
    Load, process, and save a speech dataset for training or testing models.

    Args:
        data_path (str): Path to the dataset to load (Hugging Face format).
        filename (str): Name of the file to save the processed dataset as a pickle file.
        model_name (str): Name of the Whisper model to use for feature extraction and tokenization.
        language (str): Language for the processor to configure.
        task (str): Task for the Whisper model processor (e.g., 'transcribe' or 'translate').

    Returns:
        None: The function saves the processed dataset to a pickle file.
    """
    
    # Ensure the filename is saved in the current directory
    filename = os.path.join(os.getcwd(), filename)

    # Load dataset from the specified data path, combining the train and test splits.
    data = DatasetDict()
    data["train"] = load_dataset(data_path, split="train+test", use_auth_token=True)
    
    # Remove unnecessary columns from the dataset (e.g., metadata, demographic information)
    columns_to_remove = ['accuracy', 'completeness', 'fluency', 
                         'prosodic', 'total', 'words', 'speaker', 'gender', 'age']
    
    # Split the dataset into train and test sets, keeping 20% for testing
    voice_dataset = data.remove_columns(columns_to_remove)["train"].train_test_split(test_size=0.2)

    # Initialize the Whisper processor (includes feature extractor and tokenizer)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    # Define a function to prepare and process each audio batch for input into the model
    def prepare_dataset(batch):
        # Extract audio data from the batch and compute input features using the feature extractor
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"],
                                                              sampling_rate=audio["sampling_rate"]).input_features[0]
        # Tokenize the associated text and store tokenized input ids as labels
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Cast the 'audio' column to the correct format with the desired sampling rate (16kHz)
    voice_dataset = voice_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Apply the dataset preparation function to all batches and remove unused columns
    voice_dataset = voice_dataset.map(prepare_dataset, 
                                      remove_columns=voice_dataset["train"].column_names, num_proc=1)

    # Save the processed dataset to a pickle file for later use
    with open(filename, 'wb') as f:
        pickle.dump(voice_dataset, f)
    
    # Inform the user that the dataset was successfully processed and saved
    print("Processed and saved dataset to", filename)

if __name__ == "__main__":
    # Example usage: process and save the 'speechocean762' dataset using the 'whisper-tiny' model
    load_process_and_save_dataset("mispeech/speechocean762", "data.pkl", 
                                  "openai/whisper-tiny", "English", "transcribe")
