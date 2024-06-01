from flask import Flask, render_template
import os
import subprocess
from subprocess import Popen, PIPE

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

    melody = [int(x) for x in result.stdout.replace('\n', '').split(' ')]
    chords = [int(x) for x in result.stderr.replace('\n', '').split(' ')]

    print(melody)

    return render_template('generate.html', melody=melody, chords=chords, key=key, bars=bars)

if __name__ == '__main__':
    app.run(debug=True, port=1601)