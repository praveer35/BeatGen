import os
import pty
import json
import math
import numpy as np
from subprocess import Popen, PIPE
from Bayesian_Opt_Engine import Bayesian_Opt_Engine
from new_voice_line import VoiceLineGenerator
from new_regeneration_line import VoiceLineRegenerator
#from new_rhythm import RhythmGenerator
from transformer_test import RhythmGenerator

def get_chords(key, bars):

    tmpout = os.dup(pty.STDOUT_FILENO)
    tmpin = os.dup(pty.STDIN_FILENO)
    pipefd_in = os.pipe()
    pipefd_out = os.pipe()
    os.dup2(pipefd_out[1], pty.STDOUT_FILENO)
    os.dup2(pipefd_in[0], pty.STDIN_FILENO)

    os.write(pipefd_in[1], str(bars).encode('UTF-8'))
    os.close(pipefd_in[1])

    if not os.fork():
        os.close(pipefd_out[1])
        os.close(pipefd_out[0])
        os.close(pipefd_in[0])
        os.execvp("./a.out", ["./a.out"])
    
    os.close(pipefd_out[1])
    os.close(pipefd_in[0])

    chordStr = os.read(pipefd_out[0], 1024).decode('UTF-8').strip()
    os.close(pipefd_out[0])
    os.dup2(tmpout, 1)
    os.dup2(tmpin, 0)
    os.close(tmpout)
    os.close(tmpin)

    chords = chordStr.split(' ')
    for i in range(len(chords)):
        chords[i] = int(chords[i])
    
    return chords

def get_voice_line(json_data):
    # p = Popen(['python3', 'voice_line.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    # melody_stdout_data, err = p.communicate(input=json.dumps(json_data))
    # if err: print('VOICE_LINE_ERR:', err)
    # try:
    #     return json.loads(melody_stdout_data)
    # except:
    #     return get_voice_line(json_data)
    try:
        gen = VoiceLineGenerator()
        return gen.engine(json_data)
    except:
        return get_voice_line(json_data)

    
def get_regeneration_line(json_data):
    # print('try')
    # p = Popen(['python3', 'regeneration_line.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    # melody_stdout_data, err = p.communicate(input=json.dumps(json_data))
    # if err: print('VOICE_LINE_ERR:', err)
    # print(json.loads(melody_stdout_data))
    # try:
    #     print(melody_stdout_data)
    #     return json.loads(melody_stdout_data)
    # except:
    #     return get_regeneration_line(json_data)
    try:
        regen = VoiceLineRegenerator()
        return regen.engine(json_data)
    except:
        return get_regeneration_line(json_data)

def get_arpeggio(json_data):
    chords = json_data['chords']
    json_data['chords'] = [json_data['chords'][0]]
    json_data['rhythm'] = [0.5] * 8
    arpeggio_raw = get_voice_line(json_data)['melody']
    arpeggio_mid = [arpeggio_raw[i][0] for i in range(len(arpeggio_raw))]
    arpeggio_mid *= len(chords) * 1
    # print(arpeggio_mid)
    for i in range(len(chords) * (8 * 1)):
        arpeggio_mid[i] += chords[i // (8 * 1)] - chords[0]
    # print(arpeggio_mid)
    arpeggio = [[i, (0.5 / 1)] for i in arpeggio_mid]
    json_data['chords'] = chords
    return arpeggio

def get_bass(json_data):
    chords = json_data['chords']
    pattern = json_data['pattern']
    bass = []
    for chord in chords:
        note = ((chord + 1) % 7 - 1) - 6
        if pattern == 0:
            for _ in range(4):
                bass.append([note, 0.5])
                bass.append([note - 7, 0.5])
        elif pattern == 1:
            for _ in range(4):
                bass.append([note - 7, 0.5])
                bass.append([note - 7, 0.5])
    return bass

def get_drums(json_data):
    bars = json_data['bars']
    pattern = json_data['pattern']
    drums = []
    for _ in range(bars):
        if pattern == 0:
            for _ in range(4):
                drums.append([0, 0.5])
                drums.append([-40, 0.5])
    return drums

def get_rhythm(json_data):
    # p = Popen(['python3', 'rhythm.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    # rhythm_stdout_data, _ = p.communicate(input=json.dumps(json_data))
    # return json.loads(rhythm_stdout_data)['rhythm']
    gen = RhythmGenerator()
    return gen.engine(json_data)['rhythm']

def PYTHON_TO_JS_MELODY_CONVERTER(rawMelody):
    melody = []
    track_measures = []
    measure_melody = []
    leftOffset = 0
    for note in rawMelody:
        if leftOffset % 16 == 0:
            if len(measure_melody) != 0:
                track_measures.append(measure_melody)
            measure_melody = []
        start = leftOffset
        end = leftOffset + note[1] * 4
        rank = 20 - note[0]
        melody.append([rank, start, end])
        measure_melody.append([int(rank), int(start) % 16, int(end - 1) % 16 + 1])
        leftOffset = end
    if len(measure_melody) != 0:
        track_measures.append(measure_melody)
    return track_measures

def synth_convert(measures, transpose=0):
    synth_melody = []
    c_scale_intervals = [0, 2, 3, 5, 7, 8, 10]
    for i in range(len(measures)):
        for note in measures[i]:
            note[0] = 20 - note[0]
            #print(note)
            synth_note = 57 + (note[0] // 7) * 12 + c_scale_intervals[int(note[0] % 7)] + transpose
            duration = (note[2] - note[1]) / 4
            if duration > 0:
                synth_melody.append((synth_note, 4 * i + note[1] / 4, 4 * i + note[2] / 4))
    return synth_melody

def vn(val):
    conv = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    append = (4 + math.floor(val / 7))
    note = val % 7
    return str(conv[note]) + str(append)

def optimize_input(chords, melody, n):
    eng = Bayesian_Opt_Engine(chords=chords, melody=melody)
    return eng.bayesian_optimization(n)

def batch_optimize_input(generations, n):
    eng = Bayesian_Opt_Engine(generations=generations)
    return eng.bayesian_optimization(n)

def dynamic_time_warping(melody1, melody2, w1=1, w2=1, w3=2, missing_penalty=10):
    n, m = len(melody1), len(melody2)
    D = np.full((n+1, m+1), float('inf'))
    D[0][0] = 0
    for i in range(1, n+1):
        D[i][0] = D[i-1][0] + missing_penalty
    for j in range(1, m+1):
        D[0][j] = D[0][j-1] + missing_penalty
    def dist(a, b):
        if b == None:
            return w2 * a[1] + w3 * abs(a[2])
        return w1 * abs(a[0] - b[0]) + w2 * abs(a[1] - b[1]) + w3 * abs(a[2] - b[2])
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(melody1[i-1], melody2[j-1])
            D[i][j] = min(
                D[i-1][j] + missing_penalty,
                D[i][j-1] + missing_penalty,
                D[i-1][j-1] + cost
            )
    dtw_cost = D[n][m]
    max_cost = (n + m) * missing_penalty
    normalized_difference = (dtw_cost / max_cost) * 100
    return normalized_difference

# def get_rhythm():
#     p = Popen(['python3', 'rhythm.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
#     rhythm_stdout_data, _ = p.communicate()
#     return json.loads(rhythm_stdout_data)['rhythm']
'''
data_to_arpeggiator = {
    'chords': [5, 3, 2, 6],
    'flutter': 8,
    'pitch_range': 12,
    'pitch_viscosity': 0,
    'hook_chord_boost_onchord': 5.0,
    'hook_chord_boost_2_and_6': 0.5,
    'hook_chord_boost_7': 0.5,
    'hook_chord_boost_else': 0.0,
    'nonhook_chord_boost_onchord': 5.0,
    'nonhook_chord_boost_2_and_6': 0.5,
    'nonhook_chord_boost_7': 0.5,
    'nonhook_chord_boost_else': 0.0,
    'already_played_boost': 0.01,
    'matchability_noise': 0.2,
    'hmm_bias': 0
}

print(get_arpeggio(data_to_arpeggiator))'''