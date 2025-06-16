import numpy as np
import librosa
import coremltools as ct
from transformers import AutoProcessor, AutoConfig
from tqdm import trange

# Load CoreML models
print("Compiling CoreML models (ANE compiler), this might take a while...")
encoder_mlmodel = ct.models.MLModel("kb-whisper-large-encoder.mlpackage")
decoder_mlmodel = ct.models.MLModel("kb-whisper-large-decoder.mlpackage")

# Model we have converted is the kb-whisper-large model
model_id = "KBLab/kb-whisper-large"

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Load config
config = AutoConfig.from_pretrained(model_id)

# Constants
MAX_LENGTH = config.max_target_positions
SAMPLE_RATE = 16000
EOT_TOKEN = processor.tokenizer.eos_token_id
INITIAL_TOKEN = config.decoder_start_token_id # the default tokens are suitable for transciribing swedish.

# Testing to implement chunking (needed for files larger than 30 sec)
CHUNK_LENGTH = 30.0  # seconds
CHUNK_OVERLAP = 5.0  # seconds

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def preprocess(audio):
    inputs = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    return inputs["input_features"]  # shape: (1, 128, 3000)

def encode(input_features):
    encoder_outputs = encoder_mlmodel.predict({"logmel_data": input_features})
    return encoder_outputs["output"]  # shape: (1, seq_len, d_model)

def decode(encoder_hidden_states, initial_token = INITIAL_TOKEN, timestamps=False):
    output_tokens = [initial_token]
    #output_tokens = initial_tokens.copy()
    
    for _ in range(MAX_LENGTH):
        if len(output_tokens) >= MAX_LENGTH:
            break  # Avoid overflow
        
        padded_input = np.zeros((1, MAX_LENGTH), dtype=np.int32)
        padded_input[0, :len(output_tokens)] = output_tokens

        decoder_outputs = decoder_mlmodel.predict({
            "decoder_input_ids": padded_input,
            "encoder_hidden_states": encoder_hidden_states
        })

        logits = decoder_outputs["output"]
        
        if timestamps:
            #Prevent <|notimestamps|> (ID 50359) from being picked
            logits[0, len(output_tokens) - 1, 50364] = -np.inf
        
        next_token_id = np.argmax(logits[0, len(output_tokens)-1])

        if next_token_id == EOT_TOKEN:
            break

        output_tokens.append(next_token_id)

    return output_tokens

def decode_tokens(tokens):
    print()
    return processor.tokenizer.decode(tokens, skip_special_tokens=False)

def decode_with_timestamps(token_ids):
    segments = []
    current_segment = {"start": None, "end": None, "tokens": []}

    for token in token_ids:
        if 50364 <= token <= 51863:
            # timestamp
            ts = (token - 50364) * 0.02
            if current_segment["start"] is None:
                current_segment["start"] = ts
            else:
                current_segment["end"] = ts
                # Decode tokens to text
                text = processor.tokenizer.decode(current_segment["tokens"], skip_special_tokens=True).strip()
                if text:
                    segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "text": text
                    })
                # Start a new segment
                current_segment = {"start": ts, "end": None, "tokens": []}
        else:
            # It's a wordpiece token
            current_segment["tokens"].append(token)

    return segments

def transcribe(file_path, timestamps=False):
    print("loading audio...")
    audio = load_audio(file_path)
    duration = len(audio) / SAMPLE_RATE

    chunk_samples = int(CHUNK_LENGTH * SAMPLE_RATE)
    hop_samples = int((CHUNK_LENGTH - CHUNK_OVERLAP) * SAMPLE_RATE)
    
    segments = []
    all_tokens = []
    
    t = trange(0, len(audio), hop_samples, desc='Chunks processed', leave=True)
    for offset in t:
        chunk = audio[offset : offset + chunk_samples]

        if len(chunk) < chunk_samples:
            # pad last chunk
            padding = np.zeros(chunk_samples - len(chunk))
            chunk = np.concatenate([chunk, padding])

        t.set_description("Extracting features, waveform → tensor", refresh=True)
        input_features = preprocess(chunk)
        
        t.set_description("Running encode model on input tensor", refresh=True)
        encoder_hidden_states = encode(input_features)
        
        t.set_description("Running decode predictions (token_ids)", refresh=True)
        token_ids = decode(encoder_hidden_states, timestamps=timestamps)

        if timestamps:
            t.set_description("Decoding tokens with timestamps", refresh=True)
            chunk_segments = decode_with_timestamps(token_ids)
            for seg in chunk_segments:
                seg["start"] += offset / SAMPLE_RATE
                seg["end"] += offset / SAMPLE_RATE
                segments.append(seg)
        else:
            all_tokens += token_ids

        if offset + chunk_samples >= len(audio):
            break  # reached the end

    if timestamps:
        for seg in segments:
            print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
        return segments
    else:
        t.set_description("Decoding all tokens", refresh=True)
        transcription = decode_tokens(all_tokens)
        print("Transcription:", transcription)
        return transcription


# TODO: Make proper method for running the "model" this way instead of just hard coding into the script.
file_path = "testAudio_1min.wav"  # Replace with your audio file
text = transcribe(file_path, True)
