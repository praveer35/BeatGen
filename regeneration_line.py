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
#import matplotlib.pyplot as plt

#time.sleep(2)

# data_input = {
#     'chords': [5, 4, 3, 6],
#     'new_rhythm': [[], [1.0, 1.0, 0.5, 1.0, 0.5]],
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
#     'hmm_bias': 0.0,
#     'current_line': [[3, 1.0], [1, 0.75], [2, 1.0], [0, 1.0], [-1, 0.25], [3, 0.5], [2, 1.0], [3, 1.0], [2, 1.0], [0, 0.5], [3, 1.0], [1, 0.75], [2, 1.0], [0, 1.0], [-1, 0.25], [3, 0.5], [2, 1.0], [3, 1.0], [2, 1.0], [0, 0.5]],
#     'bar': 1,
#     'isolated': True
# }

data_input = json.loads(sys.stdin.read())

# print(sys.stdin.read())

chords = data_input['chords']
new_rhythm = data_input['new_rhythm']
bars = len(chords)
pitch_range = int(data_input['pitch_range'])
pitch_viscosity = int(data_input['pitch_viscosity'])
hook_chord_boost_onchord = float(data_input['hook_chord_boost_onchord'])
hook_chord_boost_2_and_6 = float(data_input['hook_chord_boost_2_and_6'])
hook_chord_boost_7 = float(data_input['hook_chord_boost_7'])
hook_chord_boost_else = float(data_input['hook_chord_boost_else'])
nonhook_chord_boost_onchord = float(data_input['nonhook_chord_boost_onchord'])
nonhook_chord_boost_2_and_6 = float(data_input['nonhook_chord_boost_2_and_6'])
nonhook_chord_boost_7 = float(data_input['nonhook_chord_boost_7'])
nonhook_chord_boost_else = float(data_input['nonhook_chord_boost_else'])
matchability_noise = float(data_input['matchability_noise'])
already_played_boost = float(data_input['already_played_boost'])
hmm_bias = float(data_input['hmm_bias'])
current_line = data_input['current_line']
bar = int(data_input['bar'])
isolated = bool(data_input['isolated'])

rhythm = []
temp_rhythm = []
current_line_measures = []
temp_current_line_measures = []
measure_sum = 0
for i in range(len(current_line)):
    if measure_sum < 4:
        temp_rhythm.append(current_line[i][1])
        measure_sum += current_line[i][1]
        temp_current_line_measures.append(current_line[i][0])
    else:
        rhythm.append(temp_rhythm)
        current_line_measures.append(temp_current_line_measures)
        temp_rhythm = [current_line[i][1]]
        measure_sum = current_line[i][1]
        temp_current_line_measures = [current_line[i][0]]
if len(temp_rhythm) > 0:
    rhythm.append(temp_rhythm)
    current_line_measures.append(temp_current_line_measures)

for i in range(bars):
    if i == bar or i > bar and not isolated:
        rhythm[i] = new_rhythm[i]
        continue

# def bar_graph(vec, note, chord):

#     notes = []

#     for i in range(len(vec)):
#         notes.append(vn(note + i - 7))
    
#     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
#     ax.bar(notes, vec, color='black', width=0.75)
#     ax.set_title(vn(note) + ", chord=" + str(chord))
#     plt.show()


exit = 1

bayesian_chosen_probabilities = []

sensibility_index = 1
average_entropy = 0

def calculate_entropy(arr):
    entropy = 0
    for p in arr:
        entropy -= 0 if p == 0 else p * math.log2(p)
    return entropy

def update_sensibility(arr, index):
    global sensibility_index
    global average_entropy
    entropy = calculate_entropy(arr)
    average_entropy += entropy/math.log2(len(arr))
    sensibility_index *= 1 - (1 - entropy/math.log2(len(arr))) * (1 - arr[index])


def getchords(key, bars):

    tmpout = os.dup(pty.STDOUT_FILENO)
    tmpin = os.dup(pty.STDIN_FILENO)
    pipefd_in = os.pipe()
    pipefd_out = os.pipe()
    os.dup2(pipefd_out[1], pty.STDOUT_FILENO)
    os.dup2(pipefd_in[0], pty.STDIN_FILENO)

    os.write(pipefd_in[1], (str(key) + ' ' + str(bars)).encode('UTF-8'))
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

#print(chords)




#if exit: os._exit(0)


def vec_random_walk(vec, iter):
    for i in range(iter):
        for j in range(len(vec)):
            if j == 0: continue
            randnum = random.uniform(0, 1)
            if randnum >= 0.75:
                vec[j] += 1
            elif randnum < 0.25:
                vec[j] -= 1
    return vec


#vec = [0, 1, 3, 5, 7]
#vec = vec_random_walk(vec, 1)
#print(vec)

#if exit: os._exit(0)





def vn(val):
    conv = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    append = (4 + math.floor(val / 7))
    note = val % 7
    return str(conv[note]) + str(append)

def stochastize(arr):
    sum = 0
    for num in arr:
        sum += num
    for i in range(len(arr)):
        arr[i] /= sum
    return arr

def skew_distribution(arr, c=0):
    # adjusted sigmoid function S, with constraints:
    # S(0) = 0
    # S(0.5) = 0.5
    # S(1) = 1
    # S''(x) > 0 for x in [0, 0.5)
    # S''(x) = 0 for x = 0.5
    # S''(x) < 0 for x in (0.5, 1]
    if c == 0: return arr
    return stochastize([0.5 + 0.5 * (2/(1-math.exp(-c/2)) - 1) * (2/(1+math.exp(c*(0.5-x))) - 1) for x in arr])

def choose_index(arr):
    rand = random.random()
    i = 0
    while rand > 0:
        rand -= arr[i]
        i += 1
    bayesian_chosen_probabilities.append(arr[i-1])
    update_sensibility(arr, i-1)
    return i - 1

def chord_boost(note, chord):
    note %= 7
    chord += 1
    if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
        #print(str(chord) + ": " + vn(note))
        return hook_chord_boost_onchord
    if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
        return hook_chord_boost_2_and_6
    if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
        return hook_chord_boost_7
    return hook_chord_boost_else

def already_played_boost_factor(note, notes_played):
    return already_played_boost if note in notes_played else 1

def reverse_gradient_factor(last_note, inc):
    diff = 2 - last_note
    if diff == 0: return 0
    return 2 / (1 + math.pow(math.fabs(diff), -inc if diff > 0 else inc))
    #return math.fabs(1 / diff) * math.pow(math.e, inc / diff)
    #return math.fabs(1 / diff) * math.pow(math.fabs(diff), inc if diff > 0 else -inc)

def match_index(measure, chord):          # scored from 0 to 1
    score = 0
    for note in measure:
        score += chord_boost(note, chord) - 3
    return 1 / (1 + math.pow(math.e, -score / len(measure)))

def sanitize_note(markov_vector, last_note, chord, index):
    curr_note = last_note + (index - 7)
    if last_note % 7 == 5 and curr_note % 7 == 1 or last_note ^ 7 == 1 and curr_note % 7 == 5:
        markov_vector[index] = 0
    elif curr_note % 7 == 5 and chord != 2 and chord != 4 and chord != 6:
        markov_vector[index] = 0
    elif chord == 4 and curr_note % 7 == 1:
        markov_vector[index] = 0

def recalculate_markov_vector(last_note, chord, min_note, max_note):
    markov_vector = [0] * 15
    for i in range(1, 7):
        l = 7 - i
        h = 7 + i
        # NOTE: current probability distribution is linear --> make it normal
        if max_note == -4096 or (last_note - i >= max_note - pitch_range and last_note - i <= max_note):
            markov_vector[l] += (1 / i) * chord_boost(last_note - i, chord) + reverse_gradient_factor(last_note, -i)
            sanitize_note(markov_vector, last_note, chord, l)
            #print("ACCEPTED:", last_note - i, max_note)
        if min_note == -4096 or (last_note + i <= min_note + pitch_range and last_note + i >= min_note):
            markov_vector[h] += (1 / i) * chord_boost(last_note + i, chord) + reverse_gradient_factor(last_note, i)
            sanitize_note(markov_vector, last_note, chord, h)
            #print("ACCEPTED:", last_note + i, min_note)
    '''if last_note % 7 == 5:      # F
        markov_vector[7+3] = 0
        markov_vector[7-4] = 0
    if last_note % 7 == 1:      # B
        markov_vector[7+4] = 0
        markov_vector[7-3] = 0'''
    return stochastize(markov_vector)

def chord_boost2(note, chord):
    note %= 7
    chord += 1
    if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
        #print(str(chord) + ": " + vn(note))
        return nonhook_chord_boost_onchord
    if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
        return nonhook_chord_boost_2_and_6
    if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
        return nonhook_chord_boost_7
    return nonhook_chord_boost_else

def recalculate_markov_vector2(last_note, chord, delta, min_note, max_note, notes_played, note_len):
    markov_vector = [0] * 15
    for i in range(14):
        l = 7 + delta - i
        h = 7 + delta + i
        # NOTE: current probability distribution is linear --> make it normal
        if l >= 14:
            #print('ERR: l=' + str(l), file=sys.stderr)
            return [0]
        if (last_note + (l - 7) >= max_note - pitch_range or last_note + (l - 7) <= max_note) and (last_note + (l - 7) <= max_note or last_note + (l - 7) >= min_note):
            if l >= 0 and l < 14:
                markov_vector[l] += 1024 * (note_len + 1) * (((1 / ((i+1)**pitch_viscosity)) * chord_boost2(last_note + (l - 7), chord)) * already_played_boost_factor(last_note + (l - 7), notes_played))
                sanitize_note(markov_vector, last_note, chord, l)
            #print("ACCEPTED:", last_note + (l-7), max_note)
        if (last_note + (h - 7) <= min_note + pitch_range or last_note + (l - 7) >= min_note) and (last_note + (h - 7) >= min_note or last_note + (h - 7) <= max_note):
            if h < 14 and h >= 0:
                markov_vector[h] += 1024 * math.log2(note_len + 1) * (((1 / ((i+1)**pitch_viscosity)) * chord_boost2(last_note + (h - 7), chord)) * already_played_boost_factor(last_note + (h - 7), notes_played))
                sanitize_note(markov_vector, last_note, chord, h)
            #print("ACCEPTED:", last_note + (h-7), min_note)
    '''if last_note % 7 == 5:      # F
        markov_vector[7+3] = 0
        markov_vector[7-4] = 0
    if last_note % 7 == 1:      # B
        markov_vector[7+4] = 0
        markov_vector[7-3] = 0'''
    markov_vector[7] = 0
    return stochastize(markov_vector)

model = None
with open('Training/melodyhmm.doc', 'rb') as handle:
    model = pickle.load(handle)

def get_hmm_vector(datagram):
    #print(model.predict([datagram]))
    likely_final_state = model.predict([datagram])[-1]

    counts = [0] * 15

    for _ in range(20):
        index = model.sample(1, random_state=random.randint(0, 2**8), currstate=likely_final_state)[0][0][0]
        if index != 4096:
            counts[index] += 1
    if max(counts) == 0: return counts
    return stochastize(counts)

def avg_vectors(theory_vector, hmm_vector, hmm_bias):
    if len(theory_vector) != len(hmm_vector):
        print(len(theory_vector), len(hmm_vector))
    assert(len(theory_vector) == len(hmm_vector))
    avg_vector = []
    for i in range(len(theory_vector)):
        if theory_vector[i] == 0:
            avg_vector.append(0)
        else:
            avg_vector.append(theory_vector[i] * (1 - hmm_bias) + hmm_vector[i] * hmm_bias)
    return stochastize(avg_vector)


#chords = [1, 5, 6, 4]

# flutter = 4                 # how many notes in a measure

key = 1
# bars = 4

notes_played = set()

measures = []
#chords = getchords(key, bars)


def loop():

    #print("---------------------------")

    notes_played = set()

    #chords = getchords(key, bars)
    #print(chords)

    min_note = -4096
    max_note = -4096

    last_note = 2

    seed = []

    keynotes = [0] * bars

    for i in range(bars):
        if i < bar or i != bar and isolated:
            keynotes[i] = current_line_measures[i][0]
            continue
        markov_vector = recalculate_markov_vector(last_note, chords[i], min_note, max_note)
        #bar_graph(markov_vector, last_note, chords[i])
        index = choose_index(markov_vector)
        #print(index - 7)
        last_note += (index - 7)
        #print(vn(last_note), chords[i])
        keynotes[i] = last_note
        #                               print(vn(last_note), chords[i])
        #print(markov_vector)

    keynotes.append(keynotes[0])
    #print(keynotes)

    #measures = []

    FLAT_NOTES = []
    x = []

    note = 0

    for i in range(bars):
        if i < bar or i != bar and isolated:
            measures.append(current_line_measures[i])
            continue
        # print('GENERATING NEW BAR')
        last_note = keynotes[i]
        temp_notes = []
        out = ""
        temp_notes.append(last_note)
        FLAT_NOTES.append(last_note)
        x.append(len(FLAT_NOTES))
        out += vn(last_note) + " "
        notes_played.add(last_note)
        if last_note < min_note or min_note == -4096:
            min_note = last_note
        if last_note > max_note or max_note == -4096:
            max_note = last_note
        measure_flutter = len(rhythm[i])
        #print(measure_flutter)
        for j in range(measure_flutter-1):
            delta = round((keynotes[i+1] - last_note) / (measure_flutter - j - 1))
            #print(max_note, min_note)
            markov_vector2 = recalculate_markov_vector2(last_note, chords[i], delta, min_note, max_note, notes_played, rhythm[i][j+1])
            #bar_graph(markov_vector2, last_note, chords[i])
            if len(markov_vector2) == 1:
                loop()
                return
            hmm_vector = get_hmm_vector(FLAT_NOTES)
            if max(hmm_vector) == 0: hmm_vector = markov_vector2
            choice_vector = avg_vectors(markov_vector2, hmm_vector, hmm_bias)
            #index = choose_index(markov_vector2)
            index = choose_index(choice_vector)
            last_note += (index - 7)
            temp_notes.append(last_note)
            FLAT_NOTES.append(last_note)
            x.append(len(FLAT_NOTES))
            out += vn(last_note) + " "
            notes_played.add(last_note)
            if last_note < min_note:
                min_note = last_note
            if last_note > max_note:
                max_note = last_note
        #                               print(out)
        measures.append(temp_notes)
        note += 1
    
    # print(' '.join([str(x) for x in chords]), file=sys.stderr)
    # print(' '.join([str(x) for x in FLAT_NOTES]))

    #plt.plot(x, FLAT_NOTES)
    #plt.title("Note graph")
    #plt.show()
    #loop()

loop()


FLAT_NOTES = list(chain.from_iterable(measures))
FLAT_RHYTHM = list(chain.from_iterable(rhythm))

# data = {
#     'chords': ' '.join([str(x) for x in chords]),
#     'melody': ' '.join([vn(x) for x in FLAT_NOTES])
# }

if len(FLAT_NOTES) != len(FLAT_RHYTHM):
    print('notes:', len(FLAT_NOTES), file=sys.stderr)
    print('rhythm:', len(FLAT_RHYTHM), file=sys.stderr)
melody = [[FLAT_NOTES[i], FLAT_RHYTHM[i]] for i in range(len(FLAT_NOTES))]

sensibility_index = 100 * math.pow(sensibility_index, 1/len(bayesian_chosen_probabilities))
average_entropy *= 100/len(bayesian_chosen_probabilities)
confidence_percentile = 100 / (1 + math.pow(np.prod([(1-x) / x for x in bayesian_chosen_probabilities]), 1/len(bayesian_chosen_probabilities)))
geometric_mean = 100 * math.pow(math.prod(bayesian_chosen_probabilities), 1/len(bayesian_chosen_probabilities))
# print('Sensibility index:', sensibility_index)
# print('Average entropy:', average_entropy)
# print('Confidence percentile:', confidence_percentile)
# print('Geometric mean:', geometric_mean)

data = {
    'melody': melody,
    'sensibility_index': sensibility_index,
    'average_entropy': average_entropy,
    'confidence_percentile': confidence_percentile,
    'geometric_mean': geometric_mean
}

json.dump(data, sys.stdout)

#print(json.)

#print(' '.join([str(x) for x in chords]), file=sys.stderr)
#print(' '.join([str(x) for x in FLAT_NOTES]))


'''
chord = 3

for i in range(50):
    note = i % 7
    if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
        print(vn(i))
'''






"""import random

matrix = [
    [0, 0.2, 0.3, 0.5],
    [0.2, 0, 0.3, 0.5],
    [0.3, 0.3, 0, 0.4],
    [0.3, 0.5, 0.2, 0]
]

notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

offset = 0

def vn(s):
    if s == 0:
        print(notes[offset % 7])
    elif s == 1:
        print(notes[(offset + 1) % 7])
    elif s == 2:
        print(notes[(offset + 2) % 7])
    elif s == 3:
        print(notes[(offset + 4) % 7])
    if random.randint(0, 1) == 0:
        print('up')
    else:
        print('down')

def sample(n, s):
    for _ in range(n):
        rand = random.uniform(0, 1)
        k = 0
        while rand > 0:
            rand -= matrix[s][k]
            if rand < 0:
                break
            k += 1
        s = k
        vn(s)

sample(8, 0)"""