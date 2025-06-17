# KB-Whisper-CoreML
Attempting to convert the KBLab/kb-whisper-large model to CoreML
## How to use
First make sure you have python3.10+ (recommended to also use a virtual environment)
```
$pip install -r requirements.txt
```
Run the conversion script first. This will not require any additional files just an internet connection to fetch the orgininal models.

### Convert to CoreML
For kb-whisper-large:
```
$python3 convert-HF-to-CoreML.py large
```
For kb-whisper-small:
```
$python3 convert-HF-to-CoreML.py small
```

### Run the converted encoder / decoder
It will be required to have a soundfile that can be used to transcribe.

Run the large model:
```
$python3 coreml_run.py large <path_to_soundfile>
```
Run the small model:
```
$python3 coreml_run.py small <path_to_soundfile>
```

#### Timestamps
If you with for the output from the model to have timestamps you simply add timestamps as a third argument. Example:
```
$python3 coreml_run.py small <path_to_soundfile> timestamps
```
