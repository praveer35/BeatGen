from flask import Flask, render_template
import os
import subprocess
from subprocess import Popen, PIPE

import json

app = Flask(__name__)

#def get_response(filename):


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate/key=<key>&bars=<bars>')
def generate(key, bars):
    #tracks = get_response("melody.py")
    result = subprocess.run(
        ['python3', 'melody.py'],
        capture_output=True,  # Capture the output of the script
        text=True             # Get the output as string
    )

    data = json.loads(result.stdout)

    melody = [int(x) for x in data['melody'].split(' ')]
    chords = [int(x) for x in data['chords'].split(' ')]

    result = subprocess.run(
        ['python3', 'rhythm.py'],
        capture_output=True,  # Capture the output of the script
        text=True             # Get the output as string
    )

    rhythm = [float(x) for x in result.stdout.replace('\n', '').split(' ')]

    #print(melody)

    return render_template('generate.html', len=len(melody), melody=melody, rhythm=rhythm, chords=chords, key=key, bars=bars)

if __name__ == '__main__':
    pid = os.fork()
    os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)
    app.run(debug=True, port=1601)