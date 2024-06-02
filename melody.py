import random
import os
import pty
import math
import sys
import json

from itertools import chain

import numpy as np
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt

def bar_graph(vec, note, chord):

    notes = []

    for i in range(len(vec)):
        notes.append(vn(note + i - 7))
    
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
    ax.bar(notes, vec, color='black', width=0.75)
    ax.set_title(vn(note) + ", chord=" + str(chord))
    plt.show()


exit = 1

def getchords(key, bars):

    pid = os.fork()
    os.execvp("g++", ["g++", "-std=c++11", "MusicGenerator.cpp"]) if pid == 0 else os.waitpid(pid, 0)

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

def choose_index(arr):
    rand = random.random()
    i = 0
    while rand > 0:
        rand -= arr[i]
        i += 1
    return i - 1

def chord_boost(note, chord):
    note %= 7
    chord += 1
    if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
        #print(str(chord) + ": " + vn(note))
        return 5
    if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
        return 0.5
    if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
        return 0.5
    return 0

def already_played_boost(note):
    return 1.25 if note in notes_played else 1

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
    return 1 / (1 + math.pow(math.e, -score / flutter))

def recalculate_markov_vector(last_note, chord, min_note, max_note):
    markov_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 7):
        l = 7 - i
        h = 7 + i
        # NOTE: current probability distribution is linear --> make it normal
        if max_note == -4096 or (last_note - i >= max_note - 8 and last_note - i <= max_note):
            markov_vector[l] += (1 / i) * chord_boost(last_note - i, chord) + reverse_gradient_factor(last_note, -i)
            #print("ACCEPTED:", last_note - i, max_note)
        if min_note == -4096 or (last_note + i <= min_note + 8 and last_note + i >= min_note):
            markov_vector[h] += (1 / i) * chord_boost(last_note + i, chord) + reverse_gradient_factor(last_note, i)
            #print("ACCEPTED:", last_note + i, min_note)
    if last_note % 7 == 5:      # F
        markov_vector[7+3] = 0
        markov_vector[7-4] = 0
    if last_note % 7 == 1:      # B
        markov_vector[7+4] = 0
        markov_vector[7-3] = 0
    return stochastize(markov_vector)

def chord_boost2(note, chord):
    note %= 7
    chord += 1
    if (note == chord % 7 or note == (chord + 2) % 7 or note == (chord + 4) % 7):
        #print(str(chord) + ": " + vn(note))
        return 5
    if (note == (chord + 1) % 7 or note == (chord + 5) % 7):
        return 1
    if (chord == 2 or chord == 3 or chord == 5) and note == (chord + 6) % 7:
        return 1
    return 0.5

def recalculate_markov_vector2(last_note, chord, delta, min_note, max_note):
    markov_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(14):
        l = 7 + delta - i
        h = 7 + delta + i
        # NOTE: current probability distribution is linear --> make it normal
        if l >= 14:
            #print('ERR: l=' + str(l), file=sys.stderr)
            return [0]
        if last_note + (l - 7) >= max_note - 8 and last_note + (l - 7) <= max_note:
            if l >= 0: markov_vector[l] += ((1 / ((i+1)**3)) * chord_boost2(last_note + (l - 7), chord)) * already_played_boost(last_note + (l - 7))
            #print("ACCEPTED:", last_note + (l-7), max_note)
        if last_note + (h - 7) <= min_note + 8 and last_note + (h - 7) >= min_note:
            if h < 14: markov_vector[h] += ((1 / ((i+1)**3)) * chord_boost2(last_note + (h - 7), chord)) * already_played_boost(last_note + (h - 7))
            #print("ACCEPTED:", last_note + (h-7), min_note)
    if last_note % 7 == 5:      # F
        markov_vector[7+3] = 0
        markov_vector[7-4] = 0
    if last_note % 7 == 1:      # B
        markov_vector[7+4] = 0
        markov_vector[7-3] = 0
    markov_vector[7] = 0
    return stochastize(markov_vector)


#chords = [1, 5, 6, 4]

flutter = 4                 # how many notes in a measure

key = 1
bars = 4

notes_played = set()

measures = []
chords = getchords(key, bars)


def loop():

    #print("---------------------------")

    notes_played = set()

    #chords = getchords(key, bars)
    #print(chords)

    min_note = -4096
    max_note = -4096

    last_note = 2

    seed = []

    keynotes = [0] * 8

    for i in range(bars):
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

    for i in range(bars):
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
        for j in range(flutter-1):
            delta = round((keynotes[i+1] - last_note) / (flutter - j - 1))
            #print(max_note, min_note)
            markov_vector2 = recalculate_markov_vector2(last_note, chords[i], delta, min_note, max_note)
            #bar_graph(markov_vector2, last_note, chords[i])
            if len(markov_vector2) == 1:
                loop()
                return
            index = choose_index(markov_vector2)
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
    
    # print(' '.join([str(x) for x in chords]), file=sys.stderr)
    # print(' '.join([str(x) for x in FLAT_NOTES]))

    #plt.plot(x, FLAT_NOTES)
    #plt.title("Note graph")
    #plt.show()
    #loop()

loop()




matches = []

for i in range(4):
    out = "measure " + str(i) + ":"
    match_row = []
    for j in range(4):
        out += " " + str(match_index(measures[i], chords[j]))
        match_row.append(match_index(measures[i], chords[j]))
    matches.append(match_row)
    #print(out)

matchability_noise = 0.2
forward_switch = False
backward_switch = False

switch02 = True
switch13 = True

if matches[0][2] + matchability_noise > matches[2][2]:
    #print('measure 0 can repeat during measure 2')
    #print(matches[0][2], matches[2][2])
    forward_switch = True
if matches[2][0] + matchability_noise > matches[0][0]:
    #print('measure 2 can repeat during measure 0')
    #print(matches[0][2], matches[2][2])
    backward_switch = True

if forward_switch and not backward_switch:
    #print('measure 0 subbed into measure 2')
    measures[2] = measures[0]
elif not forward_switch and backward_switch:
    #print('measure 2 subbed into measure 0')
    measures[0] = measures[2]
elif forward_switch and backward_switch:
    if matches[0][0] + matches[0][2] > matches[2][0] + matches[2][2]:
        #print('measure 0 subbed into measure 2')
        measures[2] = measures[0]
    else:
        #print('measure 2 subbed into measure 0')
        measures[0] = measures[2]
else:
    switch02 = False

forward_switch = False
backward_switch = False

if matches[1][3] + matchability_noise > matches[3][3]:
    #print('measure 1 can repeat during measure 3')
    forward_switch = True
if matches[3][1] + matchability_noise > matches[1][1]:
    #print('measure 3 can repeat during measure 1')
    backward_switch = True

if forward_switch and not backward_switch:
    #print('measure 1 subbed into measure 3')
    measures[3] = measures[1]
elif not forward_switch and backward_switch:
    #print('measure 3 subbed into measure 1')
    measures[1] = measures[3]
elif forward_switch and backward_switch:
    if matches[1][1] + matches[1][3] > matches[3][1] + matches[3][3]:
        #print('measure 1 subbed into measure 3')
        measures[3] = measures[1]
    else:
        #print('measure 3 subbed into measure 1')
        measures[1] = measures[3]
else:
    switch13 = False


FLAT_NOTES = list(chain.from_iterable(measures))

data = {
    'chords': ' '.join([str(x) for x in chords]),
    'melody': ' '.join([str(x) for x in FLAT_NOTES])
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