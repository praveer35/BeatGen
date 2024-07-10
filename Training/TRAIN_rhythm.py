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
if not os.path.exists('rhythmhmm.doc'):

    input_data = []

    for i in range(1, 910):

        POP909_index = ('00' + str(i) if i < 10 else ('0' + str(i) if i < 100 else str(i)))
        POP909_midipath = 'POP909/' + POP909_index + '/' + POP909_index + '.mid'

        mididata = MidiData.MidiData(POP909_midipath)
        bps = 1000 / mididata.ms_per_beat

        notes = mididata.get_track(1).notes

        note_lens = []

        for j in range(1, len(notes)):
            note_str_prev = notes[j-1].__str__().split(' ')
            note_str_curr = notes[j].__str__().split(' ')
            time_in_sec = float(note_str_curr[2][:-1]) - float(note_str_prev[2][:-1])
            beats = round(8 * time_in_sec * bps) * 4
            if beats > 32 * 4: beats = 32 * 4
            if beats < 32 / 4: beats = 0
            note_lens.append(beats)
            #print(note_str_prev, time_in_sec, bps, round(8 * time_in_sec * bps) / 8)

        input_data.append(note_lens)
        #print(note_lens)
        #print(note_sequence)
        print('DATASET #' + str(i) + ': success.')
    # Flatten the list and reshape for HMM (needs 2D array for training)
    sequence = np.concatenate([np.array(rhythm).reshape(-1, 1) for rhythm in input_data])

    # Define the HMM model
    n_components = 3  # Number of hidden states (you may need to adjust this)
    model = hmm.CategoricalHMM(n_components=n_components, n_iter=1000)
    # Fit the model to your sequence
    model.fit(sequence)
    with open('rhythmhmm.doc', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('rhythmhmm.doc', 'rb') as handle:
        model = pickle.load(handle)

# Generate a sequence of beats from the trained HMM
num_samples = 100  # Number of samples to generate
samples, _ = model.sample(num_samples, random_state=random.randint(0, 2**8))

# Convert the samples to integers
generated_rhythms = samples.flatten().tolist()
converted_rhythms = [beat / 32 for beat in generated_rhythms]

sum = 0
final_rhythm = []
for i in range(len(converted_rhythms)):
    if sum >= 4: break
    if converted_rhythms[i] < 4 - sum:
        final_rhythm.append(converted_rhythms[i])
        sum += converted_rhythms[i]

if sum < 4: final_rhythm.append(4 - sum)

print("Generated rhythmic sequence:", converted_rhythms)
print("Final rhythmic sequence:", final_rhythm)