from flask import Flask, render_template
import os
import subprocess
from subprocess import Popen, PIPE

import json
import pty

import lib

from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

#def get_response(filename):


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate/key=<key>&bars=<bars>')
def generate(key, bars):
    #tracks = get_response("melody.py")
    chords = lib.get_chords(key, bars)
    print('chords:', chords)


    data_to_rhythm = {
        'chords': chords,
    }

    rhythm = lib.get_rhythm(data_to_rhythm)

    data_to_melody = {
        'chords': chords,
        'rhythm': rhythm,
        'flutter': 4,
        'pitch_range': 8,
        'pitch_viscosity': 4,
        'hook_chord_boost_onchord': 5.0,
        'hook_chord_boost_2_and_6': 0.5,
        'hook_chord_boost_7': 0.5,
        'hook_chord_boost_else': 0.0,
        'nonhook_chord_boost_onchord': 3,
        'nonhook_chord_boost_2_and_6': 1,
        'nonhook_chord_boost_7': 1,
        'nonhook_chord_boost_else': 0.5,
        'already_played_boost': 1.25,
        'matchability_noise': 0.1,
    }

    melody = lib.get_voice_line(data_to_melody)
    print('melody:', melody)

    data_to_arpeggiator = {
        'chords': chords,
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
    }

    arpeggio = lib.get_arpeggio(data_to_arpeggiator)
    print('arpeggio:', arpeggio)

    return render_template('generate.html', len=len(melody), melody=melody, rhythm=rhythm, arpeggio=arpeggio, chords=chords, key=key, bars=len(chords))

@app.route('/regenerate/<conditions>', methods=['GET'])
def regenerate(conditions):
    print(conditions)
    data = json.loads(conditions)
    out_data = dict()
    if data['spec'] == 'change-arpeggio':
        data_to_arpeggiator = {
            'chords': data['chords'],
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
        }
        arpeggio = lib.get_arpeggio(data_to_arpeggiator)
        out_data['arpeggio'] = arpeggio
    elif data['spec'] == 'change-rhythm':
        melody = data['melody']
    elif data['spec'] == 'change-melody':
        data_to_melody = {
            'chords': data['chords'],
            'rhythm': data['rhythm'],
            'flutter': 4,
            'pitch_range': 8,
            'pitch_viscosity': 4,
            'hook_chord_boost_onchord': 5.0,
            'hook_chord_boost_2_and_6': 0.5,
            'hook_chord_boost_7': 0.5,
            'hook_chord_boost_else': 0.0,
            'nonhook_chord_boost_onchord': 3,
            'nonhook_chord_boost_2_and_6': 1,
            'nonhook_chord_boost_7': 1,
            'nonhook_chord_boost_else': 0.5,
            'already_played_boost': 1.25,
            'matchability_noise': 0.1,
        }
        melody = lib.get_voice_line(data_to_melody)
        out_data['melody'] = melody
    elif data['spec'] == 'change-chords':
        chords = lib.get_chords(1, 4)
        data_to_rhythm = {
            'chords': chords,
        }

        rhythm = lib.get_rhythm(data_to_rhythm)

        data_to_melody = {
            'chords': chords,
            'rhythm': rhythm,
            'flutter': 4,
            'pitch_range': 8,
            'pitch_viscosity': 4,
            'hook_chord_boost_onchord': 5.0,
            'hook_chord_boost_2_and_6': 0.5,
            'hook_chord_boost_7': 0.5,
            'hook_chord_boost_else': 0.0,
            'nonhook_chord_boost_onchord': 3,
            'nonhook_chord_boost_2_and_6': 1,
            'nonhook_chord_boost_7': 1,
            'nonhook_chord_boost_else': 0.5,
            'already_played_boost': 1.25,
            'matchability_noise': 0.1,
        }

        melody = lib.get_voice_line(data_to_melody)
        print('melody:', melody)

        data_to_arpeggiator = {
            'chords': chords,
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
        }

        arpeggio = lib.get_arpeggio(data_to_arpeggiator)

        out_data['chords'] = chords
        out_data['melody'] = melody
        out_data['arpeggio'] = arpeggio

    return json.dumps(out_data)

if __name__ == '__main__':
    pid = os.fork()
    os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)
    app.run(debug=True, port=1601)