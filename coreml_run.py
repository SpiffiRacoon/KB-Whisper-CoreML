import numpy as np
import librosa
import coremltools as ct
from transformers import WhisperProcessor, AutoConfig
import soundfile as sf

# Load CoreML models
encoder_mlmodel = ct.models.MLModel("kb-whisper-large-encoder.mlpackage")
decoder_mlmodel = ct.models.MLModel("kb-whisper-large-decoder.mlpackage")

# Model we have converted is the kb-whisper-large model
model_id = "KBLab/kb-whisper-large"

# Load processor
processor = WhisperProcessor.from_pretrained(model_id)

# Load config
config = AutoConfig.from_pretrained(model_id)

# Constants
MAX_LENGTH = config.max_target_positions
SAMPLE_RATE = 16000
EOT_TOKEN = processor.tokenizer.eos_token_id
INITIAL_TOKEN = config.decoder_start_token_id

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def preprocess(audio):
    inputs = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    return inputs["input_features"]  # shape: (1, 128, 3000)

def encode(input_features):
    encoder_outputs = encoder_mlmodel.predict({"logmel_data": input_features})
    return encoder_outputs["output"]  # shape: (1, seq_len, d_model)

def decode(encoder_hidden_states, INITIAL_TOKEN):
    output_tokens = [INITIAL_TOKEN]
    
    for _ in range(MAX_LENGTH):
        padded_input = np.zeros((1, MAX_LENGTH), dtype=np.int32)
        padded_input[0, :len(output_tokens)] = output_tokens

        decoder_outputs = decoder_mlmodel.predict({
            "decoder_input_ids": padded_input,
            "encoder_hidden_states": encoder_hidden_states
        })

        logits = decoder_outputs["output"]
        next_token_id = np.argmax(logits[0, len(output_tokens)-1])

        if next_token_id == EOT_TOKEN:
            break

        output_tokens.append(next_token_id)

    return output_tokens

def decode_tokens(tokens):
    return processor.tokenizer.decode(tokens, skip_special_tokens=True)

def transcribe(file_path):
    audio = load_audio(file_path)
    input_features = preprocess(audio)
    encoder_hidden_states = encode(input_features)
    token_ids = decode(encoder_hidden_states)
    transcription = decode_tokens(token_ids)
    return transcription

# TODO: Make proper method for running the "model" this way instead of just hard coding into the script.
file_path = "testAudio_1min.wav"  # Replace with your audio file
text = transcribe(file_path)
print("Transcription:", text)
