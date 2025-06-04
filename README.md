# KB-Whisper-CoreML
Attempting to convert the KBLab/kb-whisper-large model to CoreML
## Key findings
### Input Shape
The input for the model seems to be a tensor with 128 channels and a lenght of 3000.
You can see this emulated as we create a random tensor in convertEncoder.py
```py
example_input = torch.rand(1, 128, 3000)
```
### Trouble tracing model becaus the output is a dict. (Need to look further into it)
Currently i've decided to just tell the model to not return a dict. 
```py
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="cache", return_dict=False).eval()
model.to(device)
```

This might result in the output CoreML model not working at all like the normal model.

Solutions: Try running with the strict tag disabled and see what we get from that.
```py
traced_model = torch.jit.trace(encoder, example_input, strict=False)
```
