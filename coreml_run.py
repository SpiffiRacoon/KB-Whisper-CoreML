import numpy as np
import torchaudio
from coremltools.models import MLModel
from transformers import AutoProcessor, AutoConfig
from tqdm import trange
import sys

# Testing to implement chunking (needed for files larger than 30 sec)
CHUNK_LENGTH = 30.0  # seconds
CHUNK_OVERLAP = 5.0  # seconds

def load_audio(file_path, sample_rate):
    waveform, original_sr = torchaudio.load(file_path)
    if original_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    # Convert from stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy()

def preprocess(audio, processor):
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
    return inputs["input_features"]  # shape: (1, num_mil_bins, 3000)

def encode(input_features, encoder_mlmodel):
    encoder_outputs = encoder_mlmodel.predict({"logmel_data": input_features})
    return encoder_outputs["output"]  # shape: (1, seq_len, d_model)

def decode(encoder_hidden_states, notimestamp_id, decoder_mlmodel, initial_token, eot_token, max_length=448, timestamps=False):
    output_tokens = [initial_token]
    padded_input = np.zeros((1, max_length), dtype=np.int32)
    
    for i in range(max_length):
        if i > 0 and output_tokens[-1] == eot_token:
            break

        padded_input[0, i] = output_tokens[-1]

        decoder_outputs = decoder_mlmodel.predict({
            "decoder_input_ids": padded_input,
            "encoder_hidden_states": encoder_hidden_states
        })

        logits = decoder_outputs["output"]
        
        if timestamps:
            #Prevent <|notimestamps|> from being picked
            logits[0, len(output_tokens) - 1, notimestamp_id] = -np.inf
        
        next_token_id = np.argmax(logits[0, len(output_tokens)-1])
        output_tokens.append(next_token_id)

    return output_tokens

def decode_with_timestamps(token_ids, notimestamp_id, processor):
    segments = []
    current_segment = {"start": None, "end": None, "tokens": []}

    for token in token_ids:
        if notimestamp_id <= token <= 51863:
            # timestamp
            ts = (token - notimestamp_id) * 0.02
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

def transcribe(file_path, model_id, timestamps=False, sample_rate=16000):
    # Load CoreML models
    print("Compiling CoreML models (ANE compiler), this might take a while...")
    encoder_mlmodel = MLModel((model_id+"-encoder.mlpackage"))
    decoder_mlmodel = MLModel((model_id+"-decoder.mlpackage"))
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load config
    config = AutoConfig.from_pretrained(model_id)

    # Configs
    max_length = config.max_target_positions
    eot_token = processor.tokenizer.eos_token_id
    initial_token = config.decoder_start_token_id # the default tokens are suitable for transciribing swedish.
    
    if model_id == "KBLab/kb-whisper-small":
        notimestamp_id = 50363
    else:
        notimestamp_id = 50364
    
    print("loading audio...")
    audio = load_audio(file_path, sample_rate)

    chunk_samples = int(CHUNK_LENGTH * sample_rate)
    hop_samples = int((CHUNK_LENGTH - CHUNK_OVERLAP) * sample_rate)
    
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
        input_features = preprocess(chunk, processor)
        
        t.set_description("Running encode model on input tensor", refresh=True)
        encoder_hidden_states = encode(input_features, encoder_mlmodel)
        
        t.set_description("Running decode predictions (token_ids)", refresh=True)
        token_ids = decode(encoder_hidden_states, notimestamp_id, decoder_mlmodel, 
                           initial_token, eot_token, max_length, timestamps=timestamps)

        if timestamps:
            t.set_description("Decoding tokens with timestamps", refresh=True)
            chunk_segments = decode_with_timestamps(token_ids, notimestamp_id, processor)
            for seg in chunk_segments:
                seg["start"] += offset / sample_rate
                seg["end"] += offset / sample_rate
                segments.append(seg)
        else:
            all_tokens += token_ids
    
    if timestamps:
        for seg in segments:
            print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
        return segments
    else:
        transcription = processor.tokenizer.decode(all_tokens, skip_special_tokens=False)
        print("Transcription:", transcription)
        return transcription

if __name__ == "__main__":
    modelSize = sys.argv[1]
    file_path = sys.argv[2] # your audio file
    timeArg = sys.argv[3]
    
    if timeArg == "timestamps":
        timestamps = True
    else:
        timestamps = False
    
    # Load fine-tuned model from hugging face
    if modelSize == "small":
        print("Running kb-whisper-small...")
        model_id = "KBLab/kb-whisper-small"
    elif modelSize == "large":
        print("Running kb-whisper-small...")
        model_id = "KBLab/kb-whisper-large"
    
    transcribe(file_path, model_id, timestamps)