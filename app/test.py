from transformers import pipeline

# Load the model
asr = pipeline("automatic-speech-recognition", "NbAiLabBeta/nb-whisper-medium")

#transcribe
transcription = asr("/root/projects/omnilingual-asr-service/data/transcribe_test.wav", generate_kwargs={'task': 'transcribe', 'language': 'no'})
print(transcription)
