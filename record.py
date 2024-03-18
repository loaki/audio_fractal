import soundcard as sc
import numpy as np
import scipy
import pygame

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

def get_spectral_rolloff(data, sr=48000, roll_percent=0.85):
    """
    Computes the spectral rolloff of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :param roll_percent: Percentage of energy below the spectral rolloff point
    :return: Spectral rolloff frequency
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    spectral_sum = np.sum(fft_mag)
    threshold = roll_percent * spectral_sum
    cum_sum = np.cumsum(fft_mag)
    spectral_rolloff = np.argmax(cum_sum >= threshold)
    return spectral_rolloff * (sr / len(data))

def get_spectral_centroid(data, sr=48000):
    """
    Computes the spectral centroid of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :return: Spectral centroid frequency
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    freq = np.fft.fftfreq(len(data), 1 / sr)
    
    # Ensure freq and fft_mag have the same shape
    min_len = min(len(freq), len(fft_mag))
    freq = freq[:min_len]
    fft_mag = fft_mag[:min_len]
    
    spectral_centroid = np.sum(freq * fft_mag) / np.sum(fft_mag)
    return spectral_centroid

def get_zero_crossing_rate(data):
    """
    Computes the zero crossing rate of a signal.
    :param data: Input signal
    :return: Zero crossing rate
    """
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    zcr = len(zero_crossings) / (len(data) - 1)
    return zcr

def get_harmonicity(data, sr=48000):
    """
    Computes the harmonicity of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :return: Harmonicity
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    spectral_sum = np.sum(fft_mag)
    harmonic_sum = np.sum(fft_mag[1:]) # Exclude the DC component
    harmonicity = harmonic_sum / spectral_sum
    return harmonicity

def get_spectral_flatness(data, sr=48000):
    """
    Computes the spectral flatness of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :return: Spectral flatness
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    spectral_sum = np.sum(fft_mag)
    geometric_mean = np.prod(fft_mag)**(1.0/len(fft_mag))
    spectral_flatness = spectral_sum / geometric_mean
    return spectral_flatness

def get_spectral_contrast(data, sr=48000):
    """
    Computes the spectral contrast of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :return: Spectral contrast
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    spectral_sum = np.sum(fft_mag)
    spectral_contrast = np.sum(np.diff(fft_mag)) / spectral_sum
    return spectral_contrast

def get_spectral_complexity(data, sr=48000):
    """
    Computes the spectral complexity of a signal.
    :param data: Input signal
    :param sr: Sampling rate
    :return: Spectral complexity
    """
    fft_data = np.fft.fft(data)
    fft_mag = np.abs(fft_data)
    spectral_complexity = np.sum(fft_mag > 0.01 * np.max(fft_mag)) # Threshold can be adjusted
    return spectral_complexity

def detect_kick(data, threshold=0.1, min_duration=0.01):
    """
    Detects a kick in the audio data.
    :param data: Audio data
    :param threshold: Amplitude threshold for detecting a kick
    :param min_duration: Minimum duration of a kick in seconds
    :return: True if a kick is detected, False otherwise
    """
    # Calculate the amplitude envelope
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
    
    return decay_duration < min_duration

def record_sound(speaker_name:str | None = None, sample_rate:int = 48000, record_sec:float = 0.1):
	if not speaker_name:
		speaker_name = str(sc.default_speaker().name)
	speaker = sc.get_microphone(id=speaker_name, include_loopback=True)
	with speaker.recorder(samplerate=sample_rate) as rec:
		while True:
			data = rec.record(numframes=sample_rate*record_sec)
			yield (get_amplitude(data), get_rms(data), get_zero_crossing_rate(data), get_spectral_rolloff(data), get_harmonicity(data), get_spectral_complexity(data), get_spectral_contrast(data), detect_kick(data))


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
	bar_height_e = 10
	bar_height_f = 10
	bar_height_g = 10
	bar_height_h = 10
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
		pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(120, 600 - bar_height_e - 10, 30, 600))
		pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(150, 600 - bar_height_f - 10, 30, 600))
		pygame.draw.rect(screen, (100, 50, 255), pygame.Rect(180, 600 - bar_height_g - 10, 30, 600))
		pygame.draw.rect(screen, (255, 0, 100), pygame.Rect(210, 600 - bar_height_h - 10, 30, 600))

		fps = clock.get_fps()
		fps_font = pygame.font.Font(None, 20)
		fps_text = fps_font.render(str(int(fps)), True, (200, 200, 200))
		screen.blit(fps_text, (10, 10))
		pygame.display.flip()

		# print(d)
		bar_height_a = min(10+d[0] * 7000, 600)
		bar_height_b = min(10+d[1] * 20000, 600)
		bar_height_c = min(10+d[2] * 1000, 600)
		bar_height_d = min(10+d[3] / 200, 600)
		bar_height_e = min(10+d[4] * d[4] * 100, 600)
		bar_height_f = min(10+d[5] / 20, 600)
		bar_height_g = min(10+(d[6] * 20) * (d[6] * 20), 600)
		bar_height_h = min(10+int(d[7] * 100), 600)

		clock.tick(fps)

	pygame.quit()

if __name__ == "__main__":
	demo()
