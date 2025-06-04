import torch
from transformers import AutoModelForSpeechSeq2Seq
import coremltools as ct

device = "mps:0" if torch.mps.is_available() else "cpu"
#torch_dtype = torch.float16 if torch.mps.is_available() else torch.float32
torch_dtype = torch.float32
model_id = "KBLab/kb-whisper-large"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="cache", return_dict=False).eval()
model.to(device)

# We will start by converting the encoder
encoder = model.model.encoder

# We trace the model with a random tensor input matching the shape of the usal input.
example_input = torch.rand(1, 128, 3000, dtype=torch_dtype).to(device)
traced_model = torch.jit.trace(encoder, example_input)

# Convert to Core ML program using the Unified Conversion API.
model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.macOS13 # We want to deply the model on macOS. Not IOS or watchOS devices
)

# Save the converted model.
model_from_trace.save("kb-whisper-large.mlpackage")