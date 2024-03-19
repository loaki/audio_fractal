import soundcard as sc
import numpy as np
import pygame
from collections import deque
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression

def get_amplitude(mic_data = None):
    if mic_data is None:
        return 0
    amplitude = np.max(np.abs(mic_data))
    if amplitude > 1:
        amplitude = 1
    return amplitude

def get_rms(mic_data=None):
    if mic_data is None:
        return 0
    rms = np.sqrt(np.mean(np.square(mic_data)))
    if rms > 1:
        rms = 1
    return rms

def detect_kick(data, rms_records, min_duration=0.4):
	"""
	Detects a kick in the audio data with improved accuracy.
	:param data: Audio data
	:param rms_records: RMS records for noise floor calculation
	:param min_duration: Minimum duration of a kick in seconds
	:return: True if a kick is detected, False otherwise
	"""
	# Calculate the amplitude envelope
	if len(rms_records) == 30:
		lst_sorted = np.sort(rms_records)
		noise_floor = np.percentile(lst_sorted, 5)
		# Use a dynamic threshold based on the noise floor and the data's characteristics
		threshold = np.median(noise_floor) + np.std(data) * 2 # Adjust the multiplier based on data's standard deviation
	else:
		return False	
	envelope = np.abs(np.fft.fft(data))	
	# Find the indices where the envelope exceeds the threshold
	kick_indices = np.where(envelope > threshold)[0]	
	# Check if there are enough indices to form a kick
	if len(kick_indices) < min_duration * len(data):
		return False	
	# Check if the kick has a quick decay
	decay_start = np.argmax(envelope[kick_indices])
	decay_end = np.argmin(envelope[kick_indices])
	decay_duration = (decay_end - decay_start) / len(data)

	if decay_duration < min_duration:
		print("kick!")
	return decay_duration < min_duration

def get_bass_density(data, sample_rate=48000):
	data_array = np.frombuffer(data, dtype=np.int16)
	# Perform FFT on the data
	fft_result = np.fft.fft(data_array)
	# Calculate the absolute value of the FFT result to get the magnitude
	magnitude = np.abs(fft_result)
	# Calculate the frequencies corresponding to the FFT result
	frequencies = np.fft.fftfreq(len(data_array), 1/sample_rate)
	# Filter the frequencies to get only the bass frequencies (20-150 Hz)
	bass_indices = np.where((frequencies >= 20) & (frequencies <= 150))[0]
	# Calculate the bass density by summing the magnitudes of the bass frequencies
	bass_density = np.sum(magnitude[bass_indices])
	return bass_density

def get_kick(data, bass_record):
	if len(bass_record) == 60:
		lst_sorted = np.array(bass_record)
		percent = np.percentile(lst_sorted, 80)
	else:
		return False
	# kick_indices = np.where(envelope > threshold)[0]	
	# # Check if there are enough indices to form a kick
	# if len(kick_indices) < min_duration * len(data):
	# 	return False	
	# # Check if the kick has a quick decay
	# decay_start = np.argmax(envelope[kick_indices])
	# decay_end = np.argmin(envelope[kick_indices])
	# decay_duration = (decay_end - decay_start) / len(data)

	# if decay_duration < min_duration:
	return get_bass_density(data) * 0.8 > percent


def record_sound(speaker_name:str | None = None, sample_rate:int = 48000, record_sec:float = 0.5):
	bass_records = deque(maxlen=60)
	if not speaker_name:
		speaker_name = str(sc.default_speaker().name)
	speaker = sc.get_microphone(id=speaker_name, include_loopback=True)
	with speaker.recorder(samplerate=sample_rate) as rec:
		while True:
			data = rec.record(numframes=sample_rate*record_sec)
			bass_records.append(get_bass_density(data))
			yield (get_amplitude(data), get_rms(data), get_kick(data, bass_records), get_bass_density(data))


def demo():
	pygame.init()
	width, height = 600, 600
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("dsp")
	clock = pygame.time.Clock()
	fps = 60
	bar_height_a = 10
	bar_height_b = 10
	bar_height_c = 10
	bar_height_d = 10

	for d in record_sound(record_sec=1/fps):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					pygame.quit()

		screen.fill((0, 0, 0))
		pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(0, 600 - bar_height_a - 10, 30, 600))
		pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(30, 600 - bar_height_b - 10, 30, 600))
		pygame.draw.rect(screen, (255, 0, 255), pygame.Rect(60, 600 - bar_height_c - 10, 30, 600))
		pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(90, 600 - bar_height_d - 10, 30, 600))


		fps = clock.get_fps()
		fps_font = pygame.font.Font(None, 20)
		fps_text = fps_font.render(str(int(fps)), True, (200, 200, 200))
		screen.blit(fps_text, (10, 10))
		pygame.display.flip()

		bar_height_a = min(10+d[0] * 7000, 600)
		bar_height_b = min(10+d[1] * 20000, 600)
		bar_height_c = min(10+d[2] * 300, 600)
		bar_height_d = min(10+d[3] / 1000000, 600)


		clock.tick(fps)

	pygame.quit()

if __name__ == "__main__":
	demo()
