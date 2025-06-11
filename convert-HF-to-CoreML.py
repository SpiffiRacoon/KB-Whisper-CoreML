import torch
from transformers import AutoModelForSpeechSeq2Seq
import coremltools as ct
import numpy as np

#device = "mps:0" if torch.mps.is_available() else "cpu"
#torch_dtype = torch.float16 if torch.mps.is_available() else torch.float32

#dims = {
#        'n_mels': config.num_mel_bins,
#        'n_vocab': config.vocab_size,
#        'n_audio_ctx': config.max_source_positions,
#        'n_audio_state': config.d_model,
#        'n_audio_head': config.encoder_attention_heads,
#        'n_audio_layer': config.encoder_layers,
#        'n_text_ctx': config.max_target_positions,
#        'n_text_state': config.d_model,
#        'n_text_head': config.decoder_attention_heads,
#        'n_text_layer': config.decoder_layers
#    }

## convert_encoder/decoder from https://github.com/ggml-org/whisper.cpp/models/convert-whisper-to-coreml.py

class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        self.decoder = decoder
        self.config = config

    def forward(self, decoder_input_ids, encoder_hidden_states):
        # Forward through the decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,  # set True if you later want to support cached decoding
        )
        return outputs[0]  # shape: (batch_size, seq_len, hidden_dim)

def convertToCoreML(hparams, encoder, decoder):
    # Begin tracing and converting encoder (easy)
    encoder.eval()

    # We use a dummy tensor with correct shape to trace the encoder.
    encoder_input_shape = (1, hparams.num_mel_bins, 3000)
    encoder_input_data = torch.randn(encoder_input_shape)
    traced_encoder = torch.jit.trace(encoder, encoder_input_data)

    # Encoder converted to mlpackage.
    encoder = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=encoder_input_shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16
    )
    
    encoder_output = traced_encoder(encoder_input_data)
    if isinstance(encoder_output, tuple):
        encoder_hidden = encoder_output[0]
    else:
        encoder_hidden = encoder_output  # should be (1, seq_len, d_model)
    
    decoder_input_ids = torch.randint(0, hparams.vocab_size, (1, 1), dtype=torch.long)
    
    decoder_module = DecoderWrapper(decoder, hparams)
    
    # Trace the wrapped decoder
    traced_decoder = torch.jit.trace(
        decoder_module,
        (decoder_input_ids, encoder_hidden),
        strict=False
    )
    
    decoder = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden.shape, dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13
    )


    return encoder, decoder

model_id = "KBLab/kb-whisper-large"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, use_safetensors=True, cache_dir="cache", return_dict=False)

hparams = model.config
print(hparams)

# Extract the encoder and decorder from the model
encoder = model.model.encoder
decoder = model.model.decoder

# convert encoder and decorder to CoreML
encoder, decoder = convertToCoreML(hparams, encoder, decoder)

# Save the converted model.
encoder.save("kb-whisper-large-encoder.mlpackage")
decoder.save("kb-whisper-large-decoder.mlpackage")