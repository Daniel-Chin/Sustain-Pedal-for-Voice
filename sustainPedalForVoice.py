print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
import scipy
from scipy import stats
from threading import Lock
from collections import namedtuple
from interactive import listen
try:
    from yin import yin
    from streamProfiler import StreamProfiler
    from harmonicSynth import HarmonicSynth, Harmonic
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    print(f'Missing module {module_name}. Please download at')
    print(f'https://github.com/Daniel-Chin/Python_Lib/blob/master/{module_name}.py')
    input('Press Enter to quit...')
    raise e

print('Preparing...')
PAGE_LEN = 1024
EXAMPLE_N_PAGE = 12
STABILITY_THRESHOLD = -.2
MIN_PITCH_DIFF = 1
MAIN_VOICE_THRU = True
N_HARMONICS = 60
DO_SWIPE = False

WRITE_FILE = None
# import random
# WRITE_FILE = f'demo.wav'

PEDAL_DOWN = b'l'
PEDAL_UP = b'p'
SR = 22050
DTYPE = (np.float32, pyaudio.paFloat32)
TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
SILENCE = np.zeros((PAGE_LEN, ))
PAGE_TIME = 1 / SR * PAGE_LEN
LONG_PAGE_LEN = PAGE_LEN * EXAMPLE_N_PAGE
LONG_IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, LONG_PAGE_LEN)

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()
tones = []
profiler = StreamProfiler(PAGE_LEN / SR)
sustaining = False  # So don't put a lock if you dont need it
stm_pitch = []
stm_page = []
# Short Term Memory. Len is EXAMPLE_N_PAGE

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(LONG_IMAGINARY_LADDER * freq_bin))) / LONG_PAGE_LEN

def main():
    global terminate_flag, sustaining
    print(f'Press {PEDAL_DOWN.decode()} to sustain. ')
    print(f'Press {PEDAL_UP  .decode()} to release. ')
    print('Press ESC to quit. ')
    terminateLock.acquire()
    pa = pyaudio.PyAudio()
    streamOutContainer.append(pa.open(
        format = DTYPE[1], channels = 1, rate = SR, 
        output = True, frames_per_buffer = PAGE_LEN,
    ))
    streamIn = pa.open(
        format = DTYPE[1], channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    try:
        while streamIn.is_active():
            op = listen((b'\x1b', PEDAL_DOWN, PEDAL_UP), 2, priorize_esc_or_arrow=True)
            if op is None:
                continue
            if op == b'\x1b':
                print('Esc received. Shutting down. ')
                break
            sustaining = op == PEDAL_DOWN
            print(sustaining)
    except KeyboardInterrupt:
        print('Ctrl+C received. Shutting down. ')
    finally:
        print('Releasing resources... ')
        terminate_flag = 1
        terminateLock.acquire()
        terminateLock.release()
        streamOutContainer[0].stop_stream()
        streamOutContainer[0].close()
        while streamIn.is_active():
            sleep(.1)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        pa.terminate()
        print('Resources released. ')

def summarize(list_pitch):
    pitches = np.array(list_pitch)
    slope, _, __, ___, _ = stats.linregress(np.arange(pitches.size), pitches)
    return np.mean(pitches), - abs(slope) / PAGE_TIME
    # return np.mean(pitches), np.var(pitches, ddof = 1)

def onAudioIn(in_data, sample_count, *_):
    global display_time, time_start, terminate_flag

    try:
        if terminate_flag == 1:
            terminate_flag = 2
            terminateLock.release()
            print('PA handler terminating. ')
            # Sadly, there is no way to notify main thread after returning. 
            return (None, pyaudio.paComplete)

        if sample_count > PAGE_LEN:
            print('Discarding audio page!')
            in_data = in_data[-PAGE_LEN:]

        profiler.gonna('hann')
        page = np.frombuffer(
            in_data, dtype = DTYPE[0]
        )
        hanned_page = HANN * page

        profiler.gonna('consume')
        consume(page, hanned_page)

        profiler.gonna('eat')
        [t.loop() for t in tones]

        profiler.gonna('mix')
        to_mix = [t.mix() for t in tones]
        if MAIN_VOICE_THRU:
            to_mix.append(page)
        if to_mix:
            mixed = np.sum(to_mix, 0)
        else:
            mixed = SILENCE
        streamOutContainer[0].write(mixed, PAGE_LEN)

        profiler.display(same_line=True)
        profiler.gonna('idle')
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

def consume(page, hanned_page):
    if not sustaining:
        stm_page.clear()
        stm_pitch.clear()
    f0 = yin(page, SR, PAGE_LEN)
    # fresh_pitch = np.log(f0) * 17.312340490667562 - 36.37631656229591
    fresh_pitch = np.log(f0) * 17.312340490667562
    stm_pitch.append(fresh_pitch)
    stm_page.append(page)
    if len(stm_pitch) < EXAMPLE_N_PAGE:
        tones.clear()
        return
    if len(stm_pitch) == EXAMPLE_N_PAGE + 1:
        stm_pitch.pop(0)
        stm_page.pop(0)
    pitch, stability = summarize(stm_pitch)
    # print(stability)
    if not tones or tones[-1].do_go:
        # ready for a new tone
        if stability > STABILITY_THRESHOLD:
            print('New tone, stability', stability)
            tones.append(Tone(pitch, stability, stm_page))
    else:
        # last tone not going yet
        go_pitch = tones[-1].pitch
        if abs(pitch - go_pitch) > MIN_PITCH_DIFF:
            # print('Tone goes!', np.exp(go_pitch * 0.05776226504666211))
            print('Tone goes!', go_pitch)
            tones[-1].go()
            for i, tone in enumerate(tones[:-1]):
                if abs(tone.pitch - go_pitch) < MIN_PITCH_DIFF:
                    tones.pop(i)
        else:
            if stability >= tones[-1].stability:
                print('Better tone, stability', stability)
                tones[-1] = Tone(pitch, stability, stm_page)

class Tone(HarmonicSynth):
    def __init__(self, pitch, stability, pages):
        super().__init__(
            N_HARMONICS, SR, PAGE_LEN, DTYPE[0], True, 
            DO_SWIPE, .3, 
        )
        self.do_go = False
        self.pitch = pitch
        self.stability = stability
        self.imitate(pitch, pages)

    def imitate(self, pitch, pages):
        freq = np.exp(pitch * 0.05776226504666211)
        long_page = np.concatenate(pages)
        self.ground_harmonics = [
            Harmonic(freq * i, 0)
            for i in range(1, 1 + N_HARMONICS)
        ]
        self.planned_harmonics = [
            Harmonic(
                f, 
                sft(long_page, f * LONG_PAGE_LEN / SR), 
            )
            for f in self.ground_harmonics
        ]
    
    def go(self):
        self.do_go = True
        self.eat(self.ground_harmonics)
    
    def loop(self):
        if self.do_go:
            self.eat(self.planned_harmonics)

main()
