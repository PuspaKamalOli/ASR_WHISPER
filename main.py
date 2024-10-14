import os
import sys
import pickle
import logging
import contextlib
import torch
import torchaudio
import sounddevice as sd
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Suppress logging messages from transformers and TensorFlow to avoid unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    A context manager to suppress standard output and standard error temporarily.
    This is useful when you want to suppress warnings or messages during model loading or processing.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def load_model_from_pickle(pickle_path: str):
    """
    Load a pre-trained Whisper model and processor from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file containing the model directory.
    
    Returns:
        model: The WhisperForConditionalGeneration model loaded from the directory.
        processor: The WhisperProcessor loaded from the directory.
    
    Raises:
        Exits the program with error message if loading fails.
    """
    try:
        with open(pickle_path, 'rb') as f:
            model_dir = pickle.load(f)
        # Load model and processor from the saved directory
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        processor = WhisperProcessor.from_pretrained(model_dir)
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load model from pickle: {e}")
        sys.exit(1)

def predict_from_audio(model, processor, audio_chunk: torch.Tensor, sample_rate: int = 16000) -> str:
    """
    Generate a transcription from an audio chunk using the Whisper model.
    
    Args:
        model: The Whisper model for generating the transcription.
        processor: The processor for preparing inputs and decoding outputs.
        audio_chunk (torch.Tensor): The audio data in tensor format.
        sample_rate (int, optional): The sample rate of the audio. Default is 16000 Hz.
    
    Returns:
        str: The transcription of the audio input.
    """
    # Resample audio to 16kHz if it is not already at 16000Hz
    if sample_rate != 16000:
        audio_chunk = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_chunk)
    
    # Prepare inputs for the model by extracting features from the audio
    inputs = processor(audio_chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    
    # Use the model to generate predicted IDs for transcription (without gradient tracking)
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=225)
    
    # Decode the predicted token IDs to obtain the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def audio_callback(indata, frames, time, status):
    """
    Callback function that is triggered each time a new audio block is available from the input stream.
    It processes the audio and generates a transcription using the Whisper model.
    
    Args:
        indata: The audio data input from the sound device (microphone).
        frames: The number of audio frames in the data block.
        time: Timing information for the audio block.
        status: Status information of the audio stream.
    """
    # If there are any warnings or issues in the audio stream, log a warning
    if status:
        logging.warning(f"Audio status: {status}")
    
    # Convert the incoming audio data to a PyTorch tensor for processing
    audio_chunk = torch.tensor(indata, dtype=torch.float32)
    
    # Generate and print the transcription from the audio chunk
    transcription = predict_from_audio(model, processor, audio_chunk)
    print(transcription)

if __name__ == "__main__":
    # Path to the pickle file containing the model directory
    pickle_path = './whisper_ASR_model.pkl'
    
    # Suppress output while loading the model and processor
    with suppress_stdout_stderr():
        model, processor = load_model_from_pickle(pickle_path)

    sample_rate = 16000  # Sample rate for the input audio
    duration = 3         # Duration of the audio chunks to capture in seconds

    try:
        # Open an audio input stream and listen for audio in chunks of specified duration
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, 
                            blocksize=int(sample_rate * duration)):
            print("Listening...")  # Notify the user that the model is now listening
            sd.sleep(int(duration * 1000 * 100))  # Sleep and let the callback handle audio processing
    except KeyboardInterrupt:
        # Stop listening when interrupted by the user (Ctrl+C)
        print("Stopped by user")
    except Exception as e:
        # Log any errors that occur during the audio streaming
        logging.error(f"An error occurred during audio streaming: {e}")
        sys.exit(1)
