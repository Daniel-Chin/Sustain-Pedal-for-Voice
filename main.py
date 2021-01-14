print('importing...')
import pyaudio
from time import time, sleep
import numpy as np
from threading import Lock
from collections import namedtuple
from interactive import listen
try:
    from yin import yin
except ImportError:
    print('Missing module "yin". Please download at')
    print('https://github.com/Daniel-Chin/Python_Lib/blob/master/yin.py')
    input('Press Enter to quit...')

print('Preparing...')
FRAME_LEN = 1024
EXAMPLE_N_FRAME = 10
STABILITY_THRESHOLD = ...
MIN_PITCH_DIFF = 1
MAIN_VOICE_THRU = False
CROSS_FADE = 0.04
PEDAL_DOWN = b'l'
PEDAL_UP = b'p'

SR = 44100
DTYPE = (np.float32, pyaudio.paFloat32)

FRAME_TIME = 1 / SR * FRAME_LEN
CROSS_FADE_N_SAMPLE = round(CROSS_FADE * FRAME_LEN)
FADE_IN_WINDOW = np.array([
    x / CROSS_FADE_N_SAMPLE for x in range(CROSS_FADE_N_SAMPLE)
], DTYPE[0])
FADE_OUT_WINDOW = np.flip(FADE_IN_WINDOW)
SILENCE = np.zeros((FRAME_LEN, ))

PitchFrame = namedtuple('PitchFrame', ('pitch', 'frame'))

streamOutContainer = []
display_time = 0
time_start = 0
terminate_flag = 0
terminateLock = Lock()

sustaining = False  # So don't put a lock if you dont need it
tones = []
stm = []   # Short Term Memory. Len is 1 + EXAMPLE_N_FRAME

def main():
    global terminate_flag, sustaining
    print(f'Press {PEDAL_DOWN.decode()} to sustain. ')
    print(f'Press {PEDAL_UP  .decode()} to release. ')
    terminateLock.acquire()
    pa = pyaudio.PyAudio()
    streamOutContainer.append(pa.open(
        format = DTYPE[1], channels = 1, rate = SR, 
        output = True, frames_per_buffer = FRAME_LEN,
    ))
    streamIn = pa.open(
        format = DTYPE[1], channels = 1, rate = SR, 
        input = True, frames_per_buffer = FRAME_LEN,
        stream_callback = onAudioIn, 
    )
    streamIn.start_stream()
    try:
        while streamIn.is_active():
            op = listen((b'\x1b', PEDAL_DOWN, PEDAL_UP), 2)
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
        sleep(.4)   # not perfect
        streamIn.stop_stream()
        streamIn.close()
        pa.terminate()
        print('Resources released. ')

class Tone:
    def __init__(self, pitch, stability, stm):
        self.pitch = pitch
        self.examples = stm[:]
        self.stability = stability
        self.do_go = False
        self.playhead = 0
    
    def go(self):
        # stitch
        sacrifice = self.examples.pop(-1)
        mutator = self.examples[0]
        patch = np.multiply(
            sacrifice[:CROSS_FADE_N_SAMPLE], 
            FADE_OUT_WINDOW,
        )
        patch += np.multiply(
            mutator[:CROSS_FADE_N_SAMPLE], 
            FADE_IN_WINDOW, 
        )
        mutator[:CROSS_FADE_N_SAMPLE] = patch
        self.do_go = True

def summarize(list_PitchFrame):
    pitches = np.array([x.pitch for x in list_PitchFrame])
    return np.mean(pitches), np.var(pitches, ddof = 1)

def onAudioIn(in_data, sample_count, *_):
    global display_time, time_start, terminate_flag

    try:
        if terminate_flag == 1:
            terminate_flag = 2
            terminateLock.release()
            print('PA handler terminated.')
            return (None, pyaudio.paComplete)

        if sample_count > FRAME_LEN:
            onAudioIn(in_data[:FRAME_LEN], FRAME_LEN)
            onAudioIn(in_data[FRAME_LEN:], sample_count - FRAME_LEN)

        idle_time = time() - time_start

        time_start = time()
        frame = np.frombuffer(
            in_data, dtype = DTYPE[0]
        )
        typing_time = time() - time_start

        time_start = time()
        consume(frame)
        consume_time = time() - time_start

        time_start = time()
        mix = []
        if sustaining:
            for tone in tones:
                if tone.do_go:
                    mix.append(tone.examples[tone.playhead])
                    tone.playhead = (tone.playhead + 1) % EXAMPLE_N_FRAME
        if MAIN_VOICE_THRU:
            mix.append(frame)
        if mix:
            frame_out = np.sum(mix)
        else:
            frame_out = SILENCE
        mix_time = time() - time_start

        time_start = time()
        streamOutContainer[0].write(frame_out, FRAME_LEN)
        write_time = time() - time_start

        time_start = time()
        display(
            write_time, typing_time, consume_time, mix_time, 
            display_time, 
            idle_time, 
        )
        display_time = time() - time_start

        time_start = time()
        return (None, pyaudio.paContinue)
    except:
        terminateLock.release()
        import traceback
        traceback.print_exc()
        return (None, pyaudio.paAbort)

def consume(frame):
    if not sustaining:
        return
    f0 = yin(frame, SR, FRAME_LEN)
    # fresh_pitch = np.log(f0) * 17.312340490667562 - 36.37631656229591
    fresh_pitch = np.log(f0) * 17.312340490667562
    pf = PitchFrame(fresh_pitch, frame)
    stm.append(pf)
    if len(stm) <= EXAMPLE_N_FRAME:
        tones.clear()
        return
    if len(stm) == EXAMPLE_N_FRAME + 2:
        stm.pop(0)
    pitch, variance = summarize(stm)
    stability = - variance
    print(stability)
    if tones[-1].do_go:
        # ready for a new tone
        if stability > STABILITY_THRESHOLD:
            tones.append(Tone(pitch, stability, stm))
    else:
        # last tone not going yet
        if abs(pitch - tones[-1].pitch) > MIN_PITCH_DIFF:
            tones[-1].go()
        else:
            if stability >= tones[-1].stability:
                tones[-1] = Tone(pitch, stability, stm)

TIMES = [
    'typing_time', 'consume_time', 'mix_time', 
    'write_time', 
    'display_time', 'idle_time',
]
def display(
    write_time, typing_time, consume_time, mix_time, 
    display_time, 
    idle_time, 
):
    return
    _locals = locals()
    print('', *[x[:-5] + ' {:4.0%}.    '.format(_locals[x] / FRAME_TIME) for x in TIMES])

main()
