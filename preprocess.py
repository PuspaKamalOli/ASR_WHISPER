
import os
import pickle
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def load_process_and_save_dataset(data_path: str, filename: 
    str, model_name: str, language: str, task: str) -> None:
    # Ensure the filename is saved in the current directory
    filename = os.path.join(os.getcwd(), filename)

    # Load dataset
    data = DatasetDict()
    data["train"] = load_dataset(data_path, 
                                 split="train+test", use_auth_token=True)
    
    # Remove specified columns and split dataset
    columns_to_remove = ['accuracy', 'completeness', 
                         'fluency', 'prosodic', 'total', 'words', 'speaker', 'gender', 'age']
    voice_dataset = data.remove_columns(columns_to_remove)["train"].train_test_split(test_size=0.2)

    # Initialize processor components
    processor = WhisperProcessor.from_pretrained(model_name, 
                                                 language=language, task=task)

    # Prepare dataset function
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"],
                                                              sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Cast audio column and map preparation function
    voice_dataset = voice_dataset.cast_column("audio", Audio(sampling_rate=16000))
    voice_dataset = voice_dataset.map(prepare_dataset, remove_columns=voice_dataset["train"].column_names, num_proc=1)

    # Save dataset to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(voice_dataset, f)
    
    print("Processed and saved dataset to", filename)

if __name__ == "__main__":
    load_process_and_save_dataset("mispeech/speechocean762", "data.pkl",
                                  "openai/whisper-tiny", "English", "transcribe")
