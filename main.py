from flask import Flask, render_template, request
import os
import subprocess
from subprocess import Popen, PIPE
import math
import asyncio
import fluidsynth

import json
import pty

import lib
#import db

from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

#def get_response(filename):

def vn(val):
    conv = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    append = (4 + math.floor(val / 7))
    note = val % 7
    return str(conv[note]) + str(append)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    measures = request.json['measures']
    print(type(measures))
    if type(measures) == str:
        measures = json.loads(measures)
    velocities = request.json['velocities']
    bpm = request.json['bpm']
    channel_velocities = []
    tracks = []
    # flat_melody = []
    # for measure in melody:
    #     for note in measure:
    #         flat_melody.append([20 - note[0], note[1], note[2]])
    # arpeggio = measures['arpeggio']
    # flat_arpeggio = []
    # for measure in arpeggio:
    #     for note in measure:
    #         flat_arpeggio.append([20 - note[0], note[1], note[2]])
    # chords = measures['chords']
    # flat_chords = []
    # for measure in chords:
    #     for note in measure:
    #         flat_chords.append([20 - note[0], note[1], note[2]])
    for key in measures.keys():
        synth_track = lib.synth_convert(measures[key])
        channel_velocities.append(velocities[key])
        tracks.append(synth_track)
    # synth_melody = lib.synth_convert(measures['melody'])
    # velocity_dict[synth_melody] = velocities['melody']
    # synth_chords = lib.synth_convert(measures['chords'])
    # velocity_dict[synth_chords] = velocities['chords']
    # synth_arpeggio = lib.synth_convert(measures['arpeggio'])
    # velocity_dict[synth_arpeggio] = velocities['arpeggio']
    # synth_arpeggio = lib.synth_convert(flat_arpeggio)
    # synth_chords = lib.synth_convert(flat_chords)
    print(tracks[0])
    # tracks = [
    #     synth_melody,
    #     synth_chords,
    #     synth_arpeggio
    #     # synth_arpeggio,
    #     # synth_chords
    # ]
    soundfont = "Yamaha_C3_Grand_Piano.sf2"
    synth = fluidsynth.Synth()
    #synth.delete()
    synth.start()
    print('synth started')
    sfid = synth.sfload(soundfont)

    # Select program for each channel
    for channel in range(len(tracks)):
        synth.program_select(channel, sfid, 0, 0)

    async def play_note_on_channel(synth, note, start_time, end_time, channel):
        await asyncio.sleep(start_time * 60 / bpm)
        synth.noteon(channel, note, channel_velocities[channel])
        await asyncio.sleep((end_time - start_time) * 60 / bpm)
        synth.noteoff(channel, note)

    async def play_track(synth, track, channel):
        tasks = [play_note_on_channel(synth, note, start_time, end_time, channel) for note, start_time, end_time in track]
        await asyncio.gather(*tasks)

    async def main_player():
        tasks = [play_track(synth, track, channel) for channel, track in enumerate(tracks)]
        await asyncio.gather(*tasks)
        synth.delete()

    # Run the main function
    asyncio.run(main_player())
    # sfid = synth.sfload(soundfont)
    # for channel in range(len(tracks) + 1):
    #     synth.program_select(channel, sfid, 0, 0)
    # async def play_track(synth, track, channel):
    #     for note, start_time, duration in track:
    #         await asyncio.sleep(start_time * 60/bpm)
    #         synth.noteon(channel, note, 100 if track == synth_melody else 0)
    #         await asyncio.sleep(duration * 60/bpm)
    #         synth.noteoff(channel, note)
    # async def play_chords(synth, chords):
    #     i = 0
    #     while i < len(chords):
    #         print('playing chord')
    #         synth.noteon(1, chords[i][0], 100)
    #         synth.noteon(1, chords[i+1][0], 100)
    #         synth.noteon(1, chords[i+2][0], 100)
    #         await asyncio.sleep(4 * 60/bpm)
    #         synth.noteoff(1, chords[i][0])
    #         synth.noteoff(1, chords[i+1][0])
    #         synth.noteoff(1, chords[i+2][0])
    #         i += 3
    # async def main_player():
    #     tasks = [play_track(synth, track, channel) for channel, track in enumerate(tracks)]
    #     tasks.append(play_chords(synth, synth_chords))
    #     await asyncio.gather(*tasks)
    #     synth.delete()
    # asyncio.run(main_player())
    return json.dumps({'data': 'success'})

@app.route('/generate/key=<key>&bars=<bars>', methods=['GET', 'POST'])
def generate(key, bars):
    if request.method == 'GET':
        #tracks = get_response("melody.py")
        chords = lib.get_chords(key, bars)
        print('chords:', chords)


        data_to_rhythm = {
            'notes': [-1] * 4,
        }

        rhythm = lib.get_rhythm(data_to_rhythm)
        print('rhythm:', rhythm)

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
            'hmm_bias': 0.5
        }

        melody = lib.get_voice_line(data_to_melody)
        print('melody:', melody)

        data_to_arpeggiator = {
            'chords': chords,
            'flutter': 4,
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

        arpeggio = lib.get_arpeggio(data_to_arpeggiator)
        print('arpeggio:', arpeggio)

        return render_template('new-generate.html', len=len(melody), melody=melody, rhythm=rhythm, arpeggio=arpeggio, chords=chords, key=key, bars=len(chords), vn=vn)
    elif request.method == 'POST':
        print(request.data)
        try:
            data = json.loads(request)
            #db.save_tracks(data['chords'], data['melody'], data['arpeggio'])
        except:
            print('sqlite err')
            return json.dumps({'data': 'None', 'status': 'failure'})
        return json.dumps({'data': 'None', 'status': 'success'})

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
            'hmm_bias': 0
        }
        arpeggio = lib.get_arpeggio(data_to_arpeggiator)
        out_data['arpeggio'] = arpeggio
    elif data['spec'] == 'change-rhythm':
        melody = data['melody']
        flat_rhythm = [note[1] for note in melody]
        rhythm = []
        temp_rhythm = []
        measure_sum = 0
        for i in range(len(flat_rhythm)):
            if measure_sum < 4:
                temp_rhythm.append(flat_rhythm[i])
                measure_sum += flat_rhythm[i]
            else:
                rhythm.append(temp_rhythm)
                temp_rhythm = [flat_rhythm[i]]
                measure_sum = flat_rhythm[i]
        if len(temp_rhythm) > 0:
            rhythm.append(temp_rhythm)
        note_lens = [len(measure) for measure in rhythm]
        data_to_rhythm = {
            'notes': note_lens
        }
        rhythm = lib.get_rhythm(data_to_rhythm)
        for i in range(len(melody)):
            melody[i][1] = rhythm[i]
        out_data['melody'] = melody
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
            'hmm_bias': 0.5
        }
        melody = lib.get_voice_line(data_to_melody)
        out_data['melody'] = melody
    elif data['spec'] == 'change-chords':
        chords = lib.get_chords(1, 4)
        data_to_rhythm = {
            'notes': [-1] * 4,
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
            'hmm_bias': 0.5
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
            'hmm_bias': 0
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