import sys
import os
import math
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
import pickle
import random

sys.path.append(os.getcwd() + '/MidiAnalysis')

#from MidiAnalysis import MidiEvents, MidiEventDecoder, MidiData, MidiParser, Util, Note

from MidiAnalysis import MidiData

model = None
if not os.path.exists('melodyhmm.doc'):

    input_data = []

    scale_reduce = [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7]
    for i in range(100):
        scale_reduce.append(0)

    for i in range(1, 910):

        POP909_index = ('00' + str(i) if i < 10 else ('0' + str(i) if i < 100 else str(i)))
        POP909_midipath = 'POP909/' + POP909_index + '/' + POP909_index + '.mid'

        mididata = MidiData.MidiData(POP909_midipath)

        delta_sequence = []
        note_sequence = []

        notes = mididata.get_track(1).notes
        note_sequence.append(notes[0].__str__().split(' ')[1])
        for j in range(1, len(notes)):
            note_sequence.append(notes[j].__str__().split(' ')[1])
            delta_sequence.append(7 + scale_reduce[int(math.fabs(notes[j].pitch - notes[j-1].pitch))] * (-1 if notes[j].pitch < notes[j-1].pitch else 1))

        input_data.append(delta_sequence)
        print(delta_sequence)
        #print(note_sequence)
        print('DATASET #' + str(i) + ': success.')

    # Find the length of the longest sequence
    max_len = max(len(seq) for seq in input_data)

    # Pad the sequences with a special value (e.g., -1)
    padded_sequences = np.full((len(input_data), max_len), 4096, dtype=int)
    for i, seq in enumerate(input_data):
        padded_sequences[i, :len(seq)] = seq


    X = np.array(list(padded_sequences))

    print(type(padded_sequences))


    n_components = 12
    n_iter = 100


    model = hmm.CategoricalHMM(n_components=n_components, n_iter=n_iter).fit(X)
    with open('melodyhmm.doc', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('melodyhmm.doc', 'rb') as handle:
        model = pickle.load(handle)

'''
datagram = []
for i in range(1):
    generated_sequence_length = 1
    #generated_sequence = model.sample(generated_sequence_length, random_state=random.randint(0, 2**8))[0][0][0]

    #print(generated_sequence)
    datagram.append(model.sample(generated_sequence_length, random_state=random.randint(0, 2**8))[0][0][0])
    #print(model.transmat_)
'''
datagram = [6]

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

def decide_next_note(datagram):
    if len(datagram) != 0:
        #print(model.predict([datagram]))
        likely_final_state = model.predict([datagram])[-1]

        counts = [0] * 15

        for i in range(20):
            counts[model.sample(1, random_state=random.randint(0, 2**8), currstate=likely_final_state)[0][0][0]] += 1

        stochastize(counts)
        datagram.append(choose_index(counts))
        #print(likely_final_state)
        #datagram.append(model.sample(1, random_state=random.randint(0, 2**8), currstate=likely_final_state)[0][0][0])

for i in range(8):
    decide_next_note(datagram)

print(datagram)

#print(datagram)



#print(mididata.get_track(1).notes)
#print(mididata.get_num_tracks())