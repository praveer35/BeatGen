from flask import Flask, render_template, redirect, url_for, request, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import os
import subprocess
from subprocess import Popen, PIPE
import math
import datetime
import asyncio
from midiutil import MIDIFile
import sqlite3
#import pygame
import pretty_midi
import numpy as np
from scipy.io.wavfile import write
import scipy.signal

import json
import pty

from itertools import chain

import lib
import midiutil
#import db

import fluidsynth

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

def vn(val):
    conv = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    append = (4 + math.floor(val / 7))
    note = val % 7
    return str(conv[note]) + str(append)

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
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT vector FROM weights WHERE user_id = ?', (session['user_id'],))
    x = cursor.fetchone()[0].split(' ')
    print(x)
    conn.close()
    return render_template('create.html', x=x)

@app.route('/saved', methods=['GET', 'POST'])
def saved():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        # print(request.json)
        if request.json['action'] == 'delete':
            generation_id = request.json['generation_id']
            # delete generation from generations
            cursor.execute(f'DELETE FROM generations WHERE user_id={session['user_id']} AND generation_id={generation_id}')
            # delete generation from generation_data
            cursor.execute(f'DELETE FROM generation_data WHERE generation_id={generation_id}')
            conn.commit()
        return 'success'

    cursor.execute("""
        SELECT g.generation_id, g.created, 
               (SELECT gd.generation_name 
                FROM generation_data gd 
                WHERE gd.generation_id = g.generation_id 
                LIMIT 1) AS generation_name
        FROM generations g
        WHERE g.user_id = ?
        ORDER BY created DESC;
    """, (session['user_id'],))
    # cursor.execute('SELECT created, generation_name, generation_id FROM generations ORDER BY created DESC')
    x = cursor.fetchall()
    #x = [[n[0], n[1] for n in cursor.fetchall()]
    print(x)
    conn.close()
    return render_template("saved.html", x=x)

@app.route('/track/<generation_id>', methods=['GET', 'POST'])
def track(generation_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    if request.method == 'POST':
        if 'action' in request.json and request.json['action'] == 'update':
            print(request.json['melody'])
            melody = np.array(request.json['melody'], dtype=np.float32).tobytes()
            chords = np.array(request.json['chords'], dtype=np.int16).tobytes()
            query = 'UPDATE generation_data SET track_data=? WHERE generation_id=? AND track_name=?'
            cursor.execute(query, (melody, generation_id, "melody"))
            cursor.execute(query, (chords, generation_id, "chords"))
            conn.commit()
        else:
            print(f'UPDATE generation_data SET generation_name="{request.json['generation_name']}" WHERE generation_id={generation_id}')
            cursor.execute(f'UPDATE generation_data SET generation_name="{request.json['generation_name']}" WHERE generation_id={generation_id}')
            conn.commit()
        return 'success'
    cursor.execute('SELECT track_name, generation_name, user_id, track_data FROM generation_data WHERE generation_id=' + generation_id)
    x = cursor.fetchall()
    melody = []
    chords = []
    arpeggio = []
    if (x[0][0] == 'melody'):
        melody = np.frombuffer(x[0][3], dtype=np.float32).reshape(-1, 2).tolist()
        chords = np.frombuffer(x[1][3], dtype=np.int16).tolist()
    else:
        melody = np.frombuffer(x[1][3], dtype=np.float32).reshape(-1, 2).tolist()
        chords = np.frombuffer(x[0][3], dtype=np.int16).tolist()
    
    # print(melody, chords)
    soundfont_titles = [x[:-4] for x in os.listdir("Soundfonts")]

    return render_template('new-generate.html', len=len(melody), melody=melody, arpeggio=arpeggio, chords=chords, key=1, bars=len(chords), soundfont_titles=soundfont_titles, generation_name=x[0][1], generation_id=generation_id, MODE='TRACK')

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
            midi.addNote(i, 0, note, start_time, duration, channel_velocities[i])  # Channel 0, velocity 100

    # Write the MIDI file to disk
    with open("output.mid", "wb") as output_file:
        midi.writeFile(output_file)

    # Function to convert MIDI to WAV
    command = ["fluidsynth", "-T", "wav", "-F", "output.wav", "Soundfonts/Yamaha_C3_Grand_Piano.sf2", "output.mid"]
    subprocess.run(command, check=True)

    return send_file('output.wav', mimetype='audio/wav', as_attachment=True)

    #return json.dumps({'data': 'success', 'wav_file': 'output.wav'})

# @app.route('/play', methods=['POST'])
# def play():
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'danger')
#         return redirect(url_for('login'))
#     measures = request.json['measures']
#     print(type(measures))
#     if type(measures) == str:
#         measures = json.loads(measures)
#     velocities = request.json['velocities']
#     bpm = request.json['bpm']
#     channel_velocities = []
#     tracks = []
#     for key in measures.keys():
#         synth_track = lib.synth_convert(measures[key])
#         channel_velocities.append(velocities[key])
#         tracks.append(synth_track)
#     print(tracks[0])
#     # Create a MIDI object
#     midi = MIDIFile(len(tracks))

#     # Add track names and set tempo
#     for i, track in enumerate(tracks):
#         midi.addTrackName(i, 0, f"Track {i + 1}")
#         midi.addTempo(i, 0, bpm)  # Setting the tempo to 120 BPM

#     # Add notes to the MIDI object
#     for i, track in enumerate(tracks):
#         for note, start_time, end_time in track:
#             duration = end_time - start_time
#             midi.addNote(i, 0, note, start_time, duration, channel_velocities[i])  # Channel 0, velocity 100

#     # Write the MIDI file to disk
#     with open("output.mid", "wb") as output_file:
#         midi.writeFile(output_file)
    
#     midi_data = pretty_midi.PrettyMIDI('output.mid')
#     wave = lambda t: scipy.signal.sawtooth(t, 0.5)
#     audio_data = midi_data.synthesize(wave=wave)
#     sample_rate = 44100
#     audio_data = np.float32(audio_data / np.max(np.abs(audio_data)) * 32767)
#     write('output.wav', sample_rate, audio_data)
#     print(f"Converted output.mid to output.wav")

#     # Initialize pygame mixer
#     #pygame.init()
#     #pygame.mixer.init()

#     # Load and play the MIDI file
#     #pygame.mixer.music.load('output.mid')
#     os.remove('output.mid')
#     #pygame.mixer.music.play()

#     #while pygame.mixer.music.get_busy():
#     #    pygame.time.Clock().tick(10)
#     # Run the main function
#     return send_file('output.wav', mimetype='audio/wav', as_attachment=True)
#     #return json.dumps({'data': 'success'})

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
            midi.addNote(i, 0, note, start_time, duration, channel_velocities[i])  # Channel 0, velocity 100

    # Write the MIDI file to disk
    with open("output.mid", "wb") as output_file:
        midi.writeFile(output_file)

    print("MIDI file has been created and saved as 'output.mid'.")
    return send_file('output.mid', as_attachment=True, download_name='output.mid')

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

    rhythm = list(chain.from_iterable(lib.get_rhythm(data_to_rhythm)))
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

    voice_line = lib.get_voice_line(data_to_melody)
    melody = voice_line['melody']
    print('melody:', melody)

    data_to_arpeggiator = {
        'chords': chords,
        'pitch_range': 12,
        'pitch_viscosity': 1,
        'hook_chord_boost_onchord': 5.0,
        'hook_chord_boost_2_and_6': 0.1,
        'hook_chord_boost_7': 0.0,
        'hook_chord_boost_else': 0.0,
        'nonhook_chord_boost_onchord': 5.0,
        'nonhook_chord_boost_2_and_6': 0.1,
        'nonhook_chord_boost_7': 0.0,
        'nonhook_chord_boost_else': 0.0,
        'already_played_boost': 0.11,
        'matchability_noise': 0.2,
        'hmm_bias': 0
    }

    arpeggio = lib.get_arpeggio(data_to_arpeggiator)
    print('arpeggio:', arpeggio)

    soundfont_titles = [x[:-4] for x in os.listdir("Soundfonts")]
    print(soundfont_titles)

    return render_template('new-generate.html', len=len(melody), melody=melody, rhythm=rhythm, arpeggio=arpeggio, chords=chords, key=key, bars=len(chords), soundfont_titles=soundfont_titles, sensibility_index=voice_line['sensibility_index'], average_entropy=voice_line['average_entropy'], confidence_percentile=voice_line['confidence_percentile'], geometric_mean=voice_line['geometric_mean'], MODE='GENERATE')

@app.route('/train', methods=['POST'])
def train():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT generations, vector FROM weights WHERE user_id = ?', (session['user_id'],))
    result = cursor.fetchone()
    generations = 0
    vector = '4 5.0 0.5 0.5 0.0 3 1 1 0.5 1.25'
    if result != None:
        generations = result[0]
        vector = result[1]
    print(generations)
    print(vector)

    cursor.execute('''
        UPDATE weights
        SET generations = generations + 1
        WHERE user_id = ?
    ''', (session['user_id'],))
    conn.commit()

    print(request.json['melody'])

    print('name=', request.json['generation_name'])
    cursor.execute('INSERT INTO generations (user_id, created, generation_id) VALUES (?, ?, ?)',
                   (session['user_id'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), generations))
    cursor.execute('INSERT INTO generation_data (generation_id, track_name, generation_name, user_id, track_data) VALUES (?, ?, ?, ?, ?)',
                   (generations, 'melody', request.json['generation_name'], session['user_id'], np.array(request.json['melody'], dtype=np.float32).tobytes()))
    cursor.execute('INSERT INTO generation_data (generation_id, track_name, generation_name, user_id, track_data) VALUES (?, ?, ?, ?, ?)',
                   (generations, 'chords', request.json['generation_name'], session['user_id'], np.array(request.json['chords'], dtype=np.int16).tobytes()))
    conn.commit()

    out_data = dict()
    out_data['input_vector'] = lib.optimize_input(request.json['chords'], request.json['melody'], 75)
    # out_data['input_vector'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x = out_data['input_vector']
    print(out_data)

    cursor.execute('SELECT generations, vector FROM weights WHERE user_id = ?', (session['user_id'],))
    result = cursor.fetchone()
    generations = 0
    vector = '0 0 0 0 0 0 0 0 0 0'
    if result != None:
        generations = result[0]
        vector = result[1]
    print(generations)
    print(vector)
    if generations == 0:
        cursor.execute('INSERT INTO weights (user_id, generations, vector) VALUES (?, ?, ?)', 
                    (session['user_id'], 0, '0 0 0 0 0 0 0 0 0 0'))
        conn.commit()
    vector = vector.split(' ')
    new_vector = [str((float(vector[i]) * int(generations) + float(x[i]))/(int(generations)+1)) for i in range(len(vector))]
    cursor.execute('''
        UPDATE weights
        SET vector = ?
        WHERE user_id = ?
    ''', (' '.join(new_vector), session['user_id']))
    conn.commit()

    conn.close()
    return out_data









@app.route('/regenerate', methods=['POST'])
def regenerate():
    # print(request.json)
    new_bars = 1 if request.json['isolated'] else 4 - int(request.json['bar'])
    new_rhythm = []
    for _ in range(int(request.json['bar'])):
        new_rhythm.append([])
    data_to_rhythm = {
        'notes': [-1] * new_bars,
    }
    new_rhythm += lib.get_rhythm(data_to_rhythm)
    
    print(new_rhythm)

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT vector FROM weights WHERE user_id = ?', (session['user_id'],))
    result = cursor.fetchone()
    vector = '4 5.0 0.5 0.5 0.0 3 1 1 0.5 1.25'
    if result != None:
        vector = result[0]
    vector = vector.split(' ')
    print(vector)

    data_input = {
        'chords': request.json['chords'],
        'new_rhythm': new_rhythm,
        'flutter': 4,
        'pitch_range': 8,
        'pitch_viscosity': float(vector[0]),
        'hook_chord_boost_onchord': float(vector[1]),
        'hook_chord_boost_2_and_6': float(vector[2]),
        'hook_chord_boost_7': float(vector[3]),
        'hook_chord_boost_else': float(vector[4]),
        'nonhook_chord_boost_onchord': float(vector[5]),
        'nonhook_chord_boost_2_and_6': float(vector[6]),
        'nonhook_chord_boost_7': float(vector[7]),
        'nonhook_chord_boost_else': float(vector[8]),
        'already_played_boost': float(vector[9]),
        'matchability_noise': 0.1,
        'hmm_bias': 0.0,
        'current_line': request.json['track'],
        'bar': request.json['bar'],
        'isolated': request.json['isolated']
    }
    melody = lib.get_regeneration_line(data_input)['melody']
    print(melody)
    return {'track': melody}
    #return {'track': [[0, 4], [0, 4], [0, 4], [0, 4]]}



if __name__ == '__main__':
    init_db()
    pid = os.fork()
    os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 1601)), debug=True)