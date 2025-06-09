import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights


## convert_encoder/decoder from https://github.com/ggml-org/whisper.cpp/models/convert-whisper-to-coreml.py
def convert_encoder(hparams, model, quantize=False):
    model.eval()

    input_shape = (1, hparams.num_mel_bins, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

def convert_decoder(hparams, model, quantize=False):
    model.eval()

    tokens_shape = (1,hparams.max_target_positions)
    audio_shape = (1, hparams.max_target_positions, hparams.d_model)

    audio_data = torch.randn(audio_shape)
    token_data = torch.randint(hparams.vocab_size, tokens_shape).long()
    print(token_data.shape)

    traced_model = torch.jit.trace(model, (token_data, audio_data))

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ],
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

#device = "mps:0" if torch.mps.is_available() else "cpu"
device = "cpu"
#torch_dtype = torch.float16 if torch.mps.is_available() else torch.float32
torch_dtype = torch.float32
model_id = "KBLab/kb-whisper-large"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="cache", return_dict=False).to(device)

hparams = model.config
print(hparams)

# Extract the encoder and decorder from the model
encoder = model.model.encoder
decoder = model.model.decoder

# convert encoder and decorder to CoreML
encoder = convert_encoder(hparams, encoder)
decoder = convert_decoder(hparams, decoder)

# Save the converted model.
encoder.save("kb-whisper-large-encoder.mlpackage")
decoder.save("kb-whisper-large-decoder.mlpackage")