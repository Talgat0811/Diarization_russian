from pyannote.audio import Pipeline
from pydub import AudioSegment
import nemo.collections.asr as nemo_asr
import os

# Your Hugging Face access token
auth_token = "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"

# Load the pre-trained model with authentication
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)

# Your audio file
audio_file = "/home/talgat/Desktop/myProjects/diarization/audios/msg5824308364-66907.000003.wav"

# Perform diarization
diarization = pipeline(audio_file)

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

def split_audio(audio_file, diarization):
    audio = AudioSegment.from_wav(audio_file)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = audio[turn.start * 1000:turn.end * 1000]
        segments.append((speaker, segment))
    return segments

# Нарежьте аудио на фрагменты по диаризации
segments = split_audio(audio_file, diarization)

# Открываем файл для записи результатов
with open('results.txt', 'w') as results_file:
    # Транскрибируйте каждый фрагмент и укажите говорящего
    for speaker, segment in segments:
        segment_file = f"/tmp/{speaker}.wav"
        segment.export(segment_file, format="wav")
        transcription = asr_model.transcribe([segment_file])
        result = f"{speaker}: {transcription[0]}\n"
        print(result)
        results_file.write(result)

