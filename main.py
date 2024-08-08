from flask import Flask, render_template, redirect, url_for, request, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import subprocess
from subprocess import Popen, PIPE
import math
import asyncio
import fluidsynth
from midiutil import MIDIFile
import sqlite3

import json
import pty

import lib
#import db

#from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.secret_key = 'supersecretkey'

#def get_response(filename):

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL
                      )''')
    conn.commit()
    conn.close()
    print('does it ever reach here?')

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))

        flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/create')
def create():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
    return render_template('create.html')

@app.route('/play', methods=['POST'])
def play():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
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

@app.route('/save', methods=['POST'])
def save():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
    measures = request.json['measures']
    print(type(measures))
    if type(measures) == str:
        measures = json.loads(measures)
    velocities = request.json['velocities']
    bpm = request.json['bpm']
    channel_velocities = []
    tracks = []
    for key in measures.keys():
        synth_track = lib.synth_convert(measures[key])
        channel_velocities.append(velocities[key])
        tracks.append(synth_track)
    print(tracks[0])
    # Create a MIDI object
    midi = MIDIFile(len(tracks))

    # Add track names and set tempo
    for i, track in enumerate(tracks):
        midi.addTrackName(i, 0, f"Track {i + 1}")
        midi.addTempo(i, 0, 120)  # Setting the tempo to 120 BPM

    # Add notes to the MIDI object
    for i, track in enumerate(tracks):
        for note, start_time, end_time in track:
            duration = end_time - start_time
            midi.addNote(i, 0, note, start_time, duration, 100)  # Channel 0, velocity 100

    # Write the MIDI file to disk
    with open("output.mid", "wb") as output_file:
        midi.writeFile(output_file)

    print("MIDI file has been created and saved as 'output.mid'.")
    # Run the main function
    return json.dumps({'data': 'success'})

@app.route('/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
    key = 1
    bars = 4
    #tracks = get_response("melody.py")
    chords = lib.get_chords(key, bars)
    print('chords:', chords)


    data_to_rhythm = {
        'notes': [-1] * 4,
    }

    rhythm = lib.get_rhythm(data_to_rhythm)
    print('rhythm:', rhythm)

    data_to_melody = {
        'chords': request.form.get('chords', type=str),  # Assume chords are passed as a string
        'rhythm': request.form.get('rhythm', type=str),  # Assume rhythm is passed as a string
        'pitch_range': request.form.get('pitch_range', type=float),
        'pitch_viscosity': request.form.get('pitch_viscosity', type=float),
        'hook_chord_boost_onchord': request.form.get('hook_chord_boost_onchord', type=float),
        'hook_chord_boost_2_and_6': request.form.get('hook_chord_boost_2_and_6', type=float),
        'hook_chord_boost_7': request.form.get('hook_chord_boost_7', type=float),
        'hook_chord_boost_else': request.form.get('hook_chord_boost_else', type=float),
        'nonhook_chord_boost_onchord': request.form.get('nonhook_chord_boost_onchord', type=float),
        'nonhook_chord_boost_2_and_6': request.form.get('nonhook_chord_boost_2_and_6', type=float),
        'nonhook_chord_boost_7': request.form.get('nonhook_chord_boost_7', type=float),
        'nonhook_chord_boost_else': request.form.get('nonhook_chord_boost_else', type=float),
        'already_played_boost': request.form.get('already_played_boost', type=float),
        'matchability_noise': request.form.get('matchability_noise', type=float),
        'hmm_bias': request.form.get('hmm_bias', type=float)
    }
    data_to_melody['chords'] = chords
    data_to_melody['rhythm'] = rhythm

    # data_to_melody = {
    #     'chords': chords,
    #     'rhythm': rhythm,
    #     'pitch_range': 8,
    #     'pitch_viscosity': 4,
    #     'hook_chord_boost_onchord': 5.0,
    #     'hook_chord_boost_2_and_6': 0.5,
    #     'hook_chord_boost_7': 0.5,
    #     'hook_chord_boost_else': 0.0,
    #     'nonhook_chord_boost_onchord': 3,
    #     'nonhook_chord_boost_2_and_6': 1,
    #     'nonhook_chord_boost_7': 1,
    #     'nonhook_chord_boost_else': 0.5,
    #     'already_played_boost': 1.25,
    #     'matchability_noise': 0.1,
    #     'hmm_bias': 0.5
    # }

    melody = lib.get_voice_line(data_to_melody)
    print('melody:', melody)

    data_to_arpeggiator = {
        'chords': chords,
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

    return render_template('new-generate.html', len=len(melody), melody=melody, rhythm=rhythm, arpeggio=arpeggio, chords=chords, key=key, bars=len(chords))

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
    init_db()
    pid = os.fork()
    os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)
    app.run(host='0.0.0.0', port=1601)