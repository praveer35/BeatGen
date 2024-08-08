import os
import pty
import json
import math
from subprocess import Popen, PIPE

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
    p = Popen(['python3', 'voice_line.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    melody_stdout_data, err = p.communicate(input=json.dumps(json_data))
    if err: print('VOICE_LINE_ERR:', err)
    try:
        return json.loads(melody_stdout_data)['melody']
    except:
        return get_voice_line(json_data)

def get_arpeggio(json_data):
    chords = json_data['chords']
    json_data['chords'] = [json_data['chords'][0]]
    json_data['rhythm'] = [0.5] * 8
    arpeggio_raw = get_voice_line(json_data)
    arpeggio_mid = [arpeggio_raw[i][0] for i in range(len(arpeggio_raw))]
    arpeggio_mid *= len(chords)
    print(arpeggio_mid)
    for i in range(len(chords) * 8):
        arpeggio_mid[i] += chords[i // 8] - chords[0]
    print(arpeggio_mid)
    arpeggio = [[i, 0.5] for i in arpeggio_mid]
    return arpeggio

def get_rhythm(json_data):
    p = Popen(['python3', 'rhythm.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    rhythm_stdout_data, _ = p.communicate(input=json.dumps(json_data))
    return json.loads(rhythm_stdout_data)['rhythm']

def synth_convert(measures):
    synth_melody = []
    c_scale_intervals = [0, 2, 3, 5, 7, 8, 10]
    for i in range(len(measures)):
        for note in measures[i]:
            note[0] = 20 - note[0]
            #print(note)
            synth_note = 57 + (note[0] // 7) * 12 + c_scale_intervals[note[0] % 7]
            duration = (note[2] - note[1]) / 4
            if duration > 0:
                synth_melody.append((synth_note, 4 * i + note[1] / 4, 4 * i + note[2] / 4))
    return synth_melody

def vn(val):
    conv = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    append = (4 + math.floor(val / 7))
    note = val % 7
    return str(conv[note]) + str(append)

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