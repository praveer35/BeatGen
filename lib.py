import os
import pty
import json
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
    return json.loads(melody_stdout_data)['melody']

def get_arpeggio(json_data):
    chords = json_data['chords']
    json_data['chords'] = [json_data['chords'][0]]
    json_data['rhythm'] = [0.5] * 8
    arpeggio_raw = get_voice_line(json_data)
    arpeggio_mid = [arpeggio_raw[i][0] for i in range(len(arpeggio_raw))]
    arpeggio_mid *= len(chords)
    for i in range(len(chords)):
        arpeggio_mid[i] += chords[i // 8] - chords[0]
    arpeggio = [[arpeggio_mid[i], 0.5] for i in arpeggio_mid]
    return arpeggio

def get_rhythm(json_data):
    p = Popen(['python3', 'rhythm.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    rhythm_stdout_data, _ = p.communicate(input=json.dumps(json_data))
    return json.loads(rhythm_stdout_data)['rhythm']

