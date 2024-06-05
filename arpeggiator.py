from subprocess import Popen, PIPE
import json

data_to_arpeggiator = {
    'chords': [6],
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

p = Popen(['python3', 'voice_line.py'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
stdout_data, stderr_data = p.communicate(input=json.dumps(data_to_arpeggiator))


melody = json.loads(stdout_data)['melody']

print(melody)
print(stderr_data)