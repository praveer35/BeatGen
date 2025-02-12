import random
import os
import pty
import math
import sys
import json
import pickle
from hmmlearn import hmm
import numpy as np
import time

import lib

from itertools import chain

import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt

#time.sleep(2)

# data_input = {
#     'chords': [int(x) for x in '6 4 3 2'.split(' ')], 
#     'rhythm': [1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.75, 0.25, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 0.5],
#     #'rhythm': lib.get_rhythm({'chords': [1,3,6,4]}),
#     'flutter': 4,
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
#     'hmm_bias': 0.0
# }

# data_input = json.loads(sys.stdin.read())

# print(sys.stdin.read())

# chords = data_input['chords']
# flat_rhythm = data_input['rhythm']
# bars = len(chords)
# pitch_range = int(data_input['pitch_range'])
# pitch_viscosity = int(data_input['pitch_viscosity'])
# hook_chord_boost_onchord = float(data_input['hook_chord_boost_onchord'])
# hook_chord_boost_2_and_6 = float(data_input['hook_chord_boost_2_and_6'])
# hook_chord_boost_7 = float(data_input['hook_chord_boost_7'])
# hook_chord_boost_else = float(data_input['hook_chord_boost_else'])
# nonhook_chord_boost_onchord = float(data_input['nonhook_chord_boost_onchord'])
# nonhook_chord_boost_2_and_6 = float(data_input['nonhook_chord_boost_2_and_6'])
# nonhook_chord_boost_7 = float(data_input['nonhook_chord_boost_7'])
# nonhook_chord_boost_else = float(data_input['nonhook_chord_boost_else'])
# matchability_noise = float(data_input['matchability_noise'])
# already_played_boost = float(data_input['already_played_boost'])
# hmm_bias = float(data_input['hmm_bias'])

# pitch_viscosity = None
# hook_chord_boost_onchord = None
# hook_chord_boost_2_and_6 = None
# hook_chord_boost_7 = None
# hook_chord_boost_else = None
# nonhook_chord_boost_onchord = None
# nonhook_chord_boost_2_and_6 = None
# nonhook_chord_boost_7 = None
# nonhook_chord_boost_else = None
# already_played_boost = None


#chords = [6, 4, 3, 2]
chords = [6, 1, 2, 3]
bars = len(chords)
#melody = [[4, 1.0], [2, 0.5], [-3, 1.5], [-2, 0.5], [-1, 0.5], [-5, 0.5], [-4, 1.0], [-3, 0.5], [-4, 1.0], [0, 0.75], [2, 0.25], [4, 1.0], [2, 0.5], [-3, 1.5], [-2, 0.5], [-1, 0.5], [-5, 0.5], [-4, 1.0], [-3, 0.5], [-4, 1.0], [0, 0.75], [2, 0.25]]
melody = [[7, 0.5], [4, 0.5], [2, 0.5], [1, 1.0], [2, 0.5], [3, 0.5], [2, 0.5], [4, 1.0], [3, 1.0], [4, 0.5], [2, 1.0], [4, 0.5], [7, 0.5], [4, 0.5], [2, 0.5], [1, 1.0], [2, 0.5], [3, 0.5], [2, 0.5], [4, 1.0], [3, 1.0], [4, 0.5], [2, 1.0], [4, 0.5]]



def engine(x):

    global bayesian_probabilities
    bayesian_probabilities = []



    pitch_viscosity = x[0]
    hook_chord_boost_onchord = x[1]
    hook_chord_boost_2_and_6 = x[2]
    hook_chord_boost_7 = x[3]
    hook_chord_boost_else = x[4]
    nonhook_chord_boost_onchord = x[5]
    nonhook_chord_boost_2_and_6 = x[6]
    nonhook_chord_boost_7 = x[7]
    nonhook_chord_boost_else = x[8]
    already_played_boost = x[9]

    rhythm = []
    measure_melody = []
    temp_rhythm = []
    temp_melody = []
    measure_sum = 0
    for i in range(len(melody)):
        if measure_sum < 4:
            temp_rhythm.append(melody[i][1])
            temp_melody.append(melody[i][0])
            measure_sum += melody[i][1]
        else:
            rhythm.append(temp_rhythm)
            measure_melody.append(temp_melody)
            temp_rhythm = [melody[i][1]]
            temp_melody = [melody[i][0]]
            measure_sum = melody[i][1]
    if len(temp_rhythm) > 0:
        rhythm.append(temp_rhythm)
        measure_melody.append(temp_melody)


    keynote_expectations = [measure_melody[0][0] - 2]
    for i in range(len(measure_melody)-1):
        keynote_expectations.append(measure_melody[i+1][0] - measure_melody[i][0])

    # print(measure_melody)
    # print(keynote_expectations)

    global notes_played
    notes_played = set()

    def stochastize(arr):
        sum = 0
        for num in arr:
            sum += num
        for i in range(len(arr)):
            arr[i] /= sum
        return arr

    def chord_boost(note, chord):
        note %= 7
        chord += 1
        if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
            return hook_chord_boost_onchord
        if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
            return hook_chord_boost_2_and_6
        if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
            return hook_chord_boost_7
        return hook_chord_boost_else

    def already_played_boost_factor(note):
        global notes_played
        return already_played_boost if note in notes_played else 1

    def reverse_gradient_factor(last_note, inc):
        diff = 2 - last_note
        if diff == 0: return 0
        return 2 / (1 + math.pow(math.fabs(diff), -inc if diff > 0 else inc))

    def sanitize_note(markov_vector, last_note, chord, index):
        curr_note = last_note + (index - 7)
        if last_note % 7 == 5 and curr_note % 7 == 1 or last_note ^ 7 == 1 and curr_note % 7 == 5:
            markov_vector[index] = 0
        elif curr_note % 7 == 5 and chord != 2 and chord != 4 and chord != 6:
            markov_vector[index] = 0
        elif chord == 4 and curr_note % 7 == 1:
            markov_vector[index] = 0

    def recalculate_markov_vector(last_note, chord):
        markov_vector = [0] * 15
        for i in range(1, 7):
            l = 7 - i
            h = 7 + i
            markov_vector[l] += (1 / i) * chord_boost(last_note - i, chord) + reverse_gradient_factor(last_note, -i)
            sanitize_note(markov_vector, last_note, chord, l)
            markov_vector[h] += (1 / i) * chord_boost(last_note + i, chord) + reverse_gradient_factor(last_note, i)
            sanitize_note(markov_vector, last_note, chord, h)
        return stochastize(markov_vector)

    def chord_boost2(note, chord):
        note %= 7
        chord += 1
        if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
            return nonhook_chord_boost_onchord
        if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
            return nonhook_chord_boost_2_and_6
        if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
            return nonhook_chord_boost_7
        return nonhook_chord_boost_else

    def recalculate_markov_vector2(last_note, chord, delta, note_len):
        global notes_played
        markov_vector = [0] * 15
        for i in range(14):
            l = 7 + delta - i
            h = 7 + delta + i
            if l >= 14:
                return [0]
            if l >= 0 and l < 14:
                markov_vector[l] += 1024 * (note_len + 1) * (((1 / ((i+1)**pitch_viscosity)) * chord_boost2(last_note + (l - 7), chord)) * already_played_boost_factor(last_note + (l - 7)))
                sanitize_note(markov_vector, last_note, chord, l)
            if h < 14 and h >= 0:
                markov_vector[h] += 1024 * math.log2(note_len + 1) * (((1 / ((i+1)**pitch_viscosity)) * chord_boost2(last_note + (h - 7), chord)) * already_played_boost_factor(last_note + (h - 7)))
                sanitize_note(markov_vector, last_note, chord, h)
        markov_vector[7] = 0
        return stochastize(markov_vector)

    measures = []

    def loop():

        last_note = 2

        keynotes = [0] * bars

        for i in range(bars):
            markov_vector = recalculate_markov_vector(last_note, chords[i])
            P = markov_vector[7 + keynote_expectations[i]]
            index = 7 + keynote_expectations[i]
            # print(P)
            global bayesian_probabilities
            bayesian_probabilities.append(P)
            last_note += (index - 7)
            keynotes[i] = last_note

        keynotes.append(keynotes[0])

        FLAT_NOTES = []
        x = []

        note = 0

        for i in range(bars):
            last_note = keynotes[i]
            temp_notes = []
            temp_notes.append(last_note)
            FLAT_NOTES.append(last_note)
            x.append(len(FLAT_NOTES))
            notes_played.add(last_note)
            measure_flutter = len(rhythm[i])
            for j in range(measure_flutter-1):
                delta = round((keynotes[i+1] - last_note) / (measure_flutter - j - 1))
                markov_vector2 = recalculate_markov_vector2(last_note, chords[i], delta, rhythm[i][j+1])
                if len(markov_vector2) == 1:
                    bayesian_probabilities = []
                    return loop()
                
                expected_delta = measure_melody[i][j+1] - last_note
                P = markov_vector2[7 + expected_delta]
                index = 7 + expected_delta
                # print(P)
                # if P == 0:
                    # print('delta=',expected_delta)
                    # print(i, j+1, measure_melody[i][j+1])
                bayesian_probabilities.append(P)
                last_note += (index - 7)
                temp_notes.append(last_note)
                FLAT_NOTES.append(last_note)
                x.append(len(FLAT_NOTES))
                notes_played.add(last_note)
            measures.append(temp_notes)
            note += 1
        
        geometric_mean = math.pow(np.prod(bayesian_probabilities), 1/len(bayesian_probabilities))
        #print(geometric_mean)
        confidence = 100 / (1 + math.pow(np.prod([(1-x) / x for x in bayesian_probabilities]), 1/len(bayesian_probabilities)))
        print(confidence)
        return confidence

        # print(' '.join([str(x) for x in FLAT_NOTES]))
        # plt.figure(facecolor='#003', figsize=(36,13))

        # plt.plot(x, FLAT_NOTES, color='white', linewidth=3)
        # plt.title("Note graph")

        # ax = plt.gca()
        # ax.set_facecolor('#003')  # Set the axes background to black
        # ax.spines['bottom'].set_color('lightblue')  # Axis border color
        # ax.spines['left'].set_color('lightblue')
        # ax.tick_params(axis='x', colors='lightblue')  # Tick colors
        # ax.tick_params(axis='y', colors='lightblue')
        # ax.yaxis.label.set_color('lightblue')  # Label colors
        # ax.xaxis.label.set_color('lightblue')
        # ax.title.set_color('lightblue')  # Title color
        # ax.grid(color='white', linestyle='--', linewidth=0.5)  # Grid lines
        # plt.show()

    return loop()


# x = [4, 5, 0.5, 0.5, 0, 3, 1, 1, 0.5, 1.25]

import random

max_x = None
max_output = 0

for _ in range(1000):
    x = [4, 5, random.random(), random.random(), random.random(), 3, 1, 1, random.random(), random.random() * 2]
    output = engine(x)
    if output > max_output:
        max_output = output
        max_x = [i for i in x]

print(max_x, max_output)