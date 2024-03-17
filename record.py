import soundcard as sc
import numpy as np


def get_amplitude(mic_data = None):
    if mic_data is None:
        return 0
    amplitude = np.max(np.abs(mic_data))
    if amplitude > 1:
        amplitude = 1
    return amplitude


def record_sound(speaker_name:str | None = None, sample_rate:int = 48000, record_sec:float = 0.1):
	if not speaker_name:
		speaker_name = str(sc.default_speaker().name)
	speaker = sc.get_microphone(id=speaker_name, include_loopback=True)
	with speaker.recorder(samplerate=sample_rate) as rec:
		while True:
			data = rec.record(numframes=sample_rate*record_sec)
			yield get_amplitude(data)
