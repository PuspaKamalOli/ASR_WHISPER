import os
import sys
import pickle
import logging
import contextlib
import torch
import torchaudio
import sounddevice as sd
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@contextlib.contextmanager
def suppress_stdout_stderr():
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

def load_model_from_pickle(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            model_dir = pickle.load(f)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        processor = WhisperProcessor.from_pretrained(model_dir)
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load model from pickle: {e}")
        sys.exit(1)

def predict_from_audio(model, processor, audio_chunk, sample_rate=16000):
    if sample_rate != 16000:
        audio_chunk = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_chunk)
    inputs = processor(audio_chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=225)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(f"Audio status: {status}")
    audio_chunk = torch.tensor(indata, dtype=torch.float32)
    transcription = predict_from_audio(model, processor, audio_chunk)
    print(transcription)

if __name__ == "__main__":
    pickle_path = './whisper_ASR_model.pkl'
    with suppress_stdout_stderr():
        model, processor = load_model_from_pickle(pickle_path)

    sample_rate = 16000
    duration = 3

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * duration)):
            print("Listening...")
            sd.sleep(int(duration * 1000 * 100))
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        logging.error(f"An error occurred during audio streaming: {e}")
        sys.exit(1)
