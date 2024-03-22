import soundcard as sc
import numpy as np
import pygame
from collections import deque


def get_amplitude(mic_data=None):
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


def get_bass_density(data, sample_rate=48000):
    if len(data.shape) == 2:
        data = data.flatten()
    data_array = np.frombuffer(data, dtype=np.int16)
    fft_result = np.fft.fft(data_array)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(data_array), 1 / sample_rate)
    bass_indices = np.where((frequencies >= 20) & (frequencies <= 150))[0]
    bass_density = np.sum(magnitude[bass_indices])
    return bass_density


def get_kick(data, bass_record):
    if len(bass_record) == 200:
        lst_sorted = np.array(bass_record)
        percent = np.percentile(lst_sorted, 80)
    else:
        return False
    return get_bass_density(data) * 0.9 > percent


def record_sound(
    speaker_name: str | None = None, sample_rate: int = 48000, record_sec: float = 1/30
):
    bass_records = deque(maxlen=200)
    amp_records = deque(maxlen=50)
    rms_records = deque(maxlen=50)
    if not speaker_name:
        speaker_name = str(sc.default_speaker().name)
    speaker = sc.get_microphone(id=speaker_name, include_loopback=True)
    with speaker.recorder(samplerate=sample_rate) as rec:
        while True:
            data = rec.record(numframes=sample_rate * record_sec)
            bass_records.append(get_bass_density(data))
            amp_records.append(get_amplitude(data))
            rms_records.append(get_rms(data))
            yield (
                np.median(amp_records),
                np.median(rms_records),
                get_kick(data, bass_records),
                np.median(bass_records),
            )


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

    for d in record_sound(record_sec=1 / fps):
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

        bar_height_a = min(10 + d[0] * 7000, 600)
        bar_height_b = min(10 + d[1] * 20000, 600)
        bar_height_c = min(10 + d[2] * 300, 600)
        bar_height_d = min(10 + d[3] / 1000000, 600)
        if d[2]:
            print("kick", d[0])

        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    demo()
