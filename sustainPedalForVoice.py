print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
import scipy
from scipy import stats
from threading import Lock
from collections import namedtuple
import wave
import random
try:
    from interactive import listen
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
TRACK_N_PAGE = 6
EXAMPLE_N_PAGE = 4
# STABILITY_THRESHOLD = -.3
STABILITY_THRESHOLD = -.03
MIN_PITCH_DIFF = 1
MAIN_VOICE_THRU = True
N_HARMONICS = 60
DO_SWIPE = False
EXPRESSION_SWIFT = 300000
DO_PROFILE = False
PEDAL_DELAY = .4

WRITE_FILE = None
WRITE_FILE = f'free_{random.randint(0, 99999)}.wav'
REALTIME_FEEDBACK = True

MASTER_VOLUME = .1
PEDAL_DOWN = b'l'
# PEDAL_UP = b'p'
SR = 22050
DTYPE = (np.int32, pyaudio.paInt32)
TWO_PI = np.pi * 2
HANN = scipy.signal.get_window('hann', PAGE_LEN, True)
SILENCE = np.zeros((PAGE_LEN, ))
PAGE_TIME = 1 / SR * PAGE_LEN
LONG_PAGE_LEN = PAGE_LEN * EXAMPLE_N_PAGE
LONG_IMAGINARY_LADDER = np.linspace(0, TWO_PI * 1j, LONG_PAGE_LEN)
LONG_HANN = scipy.signal.get_window('hann', LONG_PAGE_LEN, True)
E_SWIFT_PER_PAGE = PAGE_TIME * EXPRESSION_SWIFT

streamOutContainer = []
terminate_flag = 0
terminateLock = Lock()
tones = []
out_tones = []
profiler = StreamProfiler(PAGE_LEN / SR, DO_PROFILE)
sustaining = False  # So don't put a lock if you dont need it
stm_pitch = []
stm_page = []
last_pedal = 0
# Short Term Memory. Len is TRACK_N_PAGE

def summarize(list_pitch):
    pitches = np.array(list_pitch)
    # slope, _, __, ___, _ = stats.linregress(np.arange(pitches.size), pitches)
    # return np.mean(pitches), - abs(slope) / PAGE_TIME
    return np.mean(pitches), - np.var(pitches, ddof = 1)

if DO_PROFILE:
    _print = print
    def print(*a, **k):
        _print()
        _print(*a, **k)

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(LONG_IMAGINARY_LADDER * freq_bin))) / LONG_PAGE_LEN

def main():
    global terminate_flag, sustaining, f, last_pedal
    print(f'Hold {PEDAL_DOWN.decode()} to sustain. ')
    # print(f'Press {PEDAL_UP  .decode()} to release. ')
    print('Press ESC to quit. ')
    terminateLock.acquire()
    pa = pyaudio.PyAudio()
    if REALTIME_FEEDBACK:
        streamOutContainer.append(pa.open(
            format = DTYPE[1], channels = 1, rate = SR, 
            output = True, frames_per_buffer = PAGE_LEN,
        ))
    if WRITE_FILE is not None:
        f = wave.open(WRITE_FILE, 'wb')
        f.setnchannels(1)
        f.setsampwidth(4)
        f.setframerate(SR)
    streamIn = pa.open(
        format = DTYPE[1], channels = 1, rate = SR, 
        input = True, frames_per_buffer = PAGE_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    try:
        while streamIn.is_active():
            op = listen((b'\x1b', PEDAL_DOWN), PEDAL_DELAY, priorize_esc_or_arrow=True)
            if op == b'\x1b':
                print('Esc received. Shutting down. ')
                break
            if sustaining:
                if op is None:
                    if time() - last_pedal > PEDAL_DELAY:
                        sustaining = False
                        print('Pedal up!')
                else:
                    last_pedal = max(time(), last_pedal)
            else:
                if op == PEDAL_DOWN:
                    sustaining = True
                    last_pedal = time() + .7
                    print('Pedal down!')
    except KeyboardInterrupt:
        print('Ctrl+C received. Shutting down. ')
    finally:
        print('Releasing resources... ')
        terminate_flag = 1
        terminateLock.acquire()
        terminateLock.release()
        if REALTIME_FEEDBACK:
            streamOutContainer[0].stop_stream()
            streamOutContainer[0].close()
        if WRITE_FILE is not None:
            f.close()
        while streamIn.is_active():
            sleep(.1)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        pa.terminate()
        print('Resources released. ')

def calcExpression(page):
    return np.sqrt(np.sum(scipy.signal.periodogram(page, SR)[1]) / page.size)

def onAudioIn(in_data, sample_count, *_):
    global terminate_flag

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

        profiler.gonna('power')
        page = np.frombuffer(
            in_data, dtype = DTYPE[0]
        )
        expression = calcExpression(page)

        consume(page)

        profiler.gonna('eat')
        [t.eatPlan() for t in tones]
        [t.eatPlan() for t in out_tones]

        profiler.gonna('update')
        [t.update(expression) for t in tones]
        [t.update(0) for t in out_tones]
        for i, t in reversed([*enumerate(out_tones)]):
            if t.scale == 0:
                out_tones.pop(i)
            else:
                t.update(0)

        profiler.gonna('mix')
        to_mix = [t.mix() for t in [*tones, *out_tones]]
        if MAIN_VOICE_THRU:
            to_mix.append(page)
        if to_mix:
            mixed = np.round(np.sum(to_mix, 0) * MASTER_VOLUME).astype(DTYPE[0])
        else:
            mixed = SILENCE
        if REALTIME_FEEDBACK:
            streamOutContainer[0].write(mixed, PAGE_LEN)
        if WRITE_FILE is not None:
            f.writeframes(mixed)

        profiler.display(same_line=True)
        profiler.gonna('idle')
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

def consume(page):
    if not sustaining:
        stm_page.clear()
        stm_pitch.clear()
    profiler.gonna('yin')
    f0 = yin(page * HANN, SR, PAGE_LEN)
    # fresh_pitch = np.log(f0) * 17.312340490667562 - 36.37631656229591
    fresh_pitch = np.log(f0) * 17.312340490667562
    stm_pitch.append(fresh_pitch)
    stm_page.append(page)
    if len(stm_pitch) < TRACK_N_PAGE:
        if tones:
            out_tones.extend(tones)
            tones.clear()
        return
    if len(stm_pitch) == TRACK_N_PAGE + 1:
        stm_pitch.pop(0)
        stm_page.pop(0)
    profiler.gonna('summ')
    pitch, stability = summarize(stm_pitch)
    # print(fresh_pitch, pitch)
    # print(stability)
    if not tones or tones[-1].do_go:
        # ready for a new tone
        if stability > STABILITY_THRESHOLD:
            freq = np.exp(pitch * 0.05776226504666211)
            if freq > 50 and freq < 1600:
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
                    out_tones.append(tones.pop(i))
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
        self.scale = 0
        self.original_expression = None
        self.example = pages[:EXAMPLE_N_PAGE]

    def imitate(self):
        profiler.gonna('concat')
        freq = np.exp(self.pitch * 0.05776226504666211)
        long_page = np.concatenate(self.example)
        profiler.gonna('sft')
        self.planned_harmonics = [
            Harmonic(
                freq * i, 
                sft(long_page * LONG_HANN, freq * i * LONG_PAGE_LEN / SR), 
            )
            for i in range(1, 1 + N_HARMONICS)
        ]
    
    def go(self):
        self.imitate()
        profiler.gonna('go')
        self.do_go = True
        self.scale = 1
        self.eatPlan()
        self.eatPlan()
        self.original_expression = calcExpression(self.mix())
        # print(self.original_expression)
        self.scale = 0
        self.eatPlan()

    def eatPlan(self):
        if self.do_go:
            self.eat([
                Harmonic(f, m * self.scale) 
                for f, m in self.planned_harmonics
            ])

    def update(self, target_expression):
        if self.do_go:
            now_expression = self.original_expression * self.scale
            if target_expression > now_expression:
                self.scale = min(
                    now_expression + E_SWIFT_PER_PAGE, 
                    target_expression, 
                ) / self.original_expression
            if target_expression < now_expression:
                self.scale = max(
                    now_expression - E_SWIFT_PER_PAGE, 
                    target_expression, 
                ) / self.original_expression
            # print(self.scale)

main()
