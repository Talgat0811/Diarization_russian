from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

# Your Hugging Face access token
auth_token = "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"

# Load the pre-trained model with authentication
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=auth_token)

# Your audio file
audio_file = "/home/talgat/Desktop/myProjects/diarization/audios/msg5824308364-66907.000003.wav"

# Perform diarization
diarization = pipeline(audio_file)

# Load the audio file using pydub
audio = AudioSegment.from_wav(audio_file)

# Create a directory to save the audio segments
output_dir = os.path.expanduser("~/Desktop/myProjects/diarization/audio_segments")
os.makedirs(output_dir, exist_ok=True)

# Split the audio and save segments
for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
    start = turn.start * 1000  # pydub works in milliseconds
    end = turn.end * 1000
    segment = audio[start:end]
    segment_file = os.path.join(output_dir, f"speaker_{speaker}_segment_{i}.wav")
    segment.export(segment_file, format="wav")
    print(f"Saved {segment_file}")

print("All segments saved successfully.")
