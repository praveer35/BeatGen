import random
import os
import pty
import math
import sys
import json

from itertools import chain

import numpy as np
import pickle
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import matplotlib.pyplot as plt

class RhythmGenerator:
    def engine(self, data_input):
        # data_input = json.loads(input())
        notes = data_input['notes']
        #notes = [-1]
        bars = len(notes)
        #bars = 4


        #FLAT_NOTES = list(chain.from_iterable(measures))

        # rhythm = []
        # for j in range(bars):
        #     choice = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        #     for i in range(4):
        #         removal_index = random.randint(1, 7 - i)
        #         del choice[removal_index]
        #     rhythm_row = []
        #     for i in range(4):
        #         rhythm_row.append(choice[i + 1] - choice[i])
        #     rhythm.append(rhythm_row)

        # rhythm_out = list(chain.from_iterable(rhythm))

        # data = {
        #     'rhythm': list(chain.from_iterable(rhythm))
        # }

        with open('Training/rhythmhmm.doc', 'rb') as handle:
            model = pickle.load(handle)

        final_rhythm = []
        temp_rhythm = []
        for i in range(bars):
            if notes[i] and notes[i] != -1:
                partitions = list(random.sample(range(1, bars * 4), notes[i] - 1))
                partitions.append(bars * 4)
                partitions.append(0)
                partitions.sort()

                #print(partitions)

                for j in range(notes[i]):
                    temp_rhythm.append((partitions[j+1] - partitions[j]) * 0.25)

            else:
                num_samples = 100  # Number of samples to generate
                samples, _ = model.sample(num_samples, random_state=random.randint(0, 2**8))

                # Convert the samples to integers
                generated_rhythms = samples.flatten().tolist()
                converted_rhythms = [beat / 16 for beat in generated_rhythms]

                sum = 0
                for j in range(len(converted_rhythms)):
                    if sum >= 4: break
                    if converted_rhythms[j] == 0:
                        continue
                    if converted_rhythms[j] < 4 - sum:
                        if converted_rhythms[j] == 0.75:
                            converted_rhythms[j] = 0.5 if random.random() < 0.5 else 1.0
                        elif converted_rhythms[j] == 0.25:
                            converted_rhythms[j] = 0.5
                        temp_rhythm.append(converted_rhythms[j])
                        sum += converted_rhythms[j]

                if sum < 4: temp_rhythm.append(4 - sum)

            final_rhythm.append(temp_rhythm)
            temp_rhythm = []

        if len(temp_rhythm) > 0:
            final_rhythm.append(temp_rhythm)

        if len(final_rhythm) == 4:
            final_rhythm[2] = final_rhythm[0]

        data = {
        #'rhythm': list(chain.from_iterable(final_rhythm)),
        'rhythm': final_rhythm
        }

        return data

# json.dump(data, sys.stdout)

#print(rhythm_out)

#data = {
#    'rhythm': ' '.join([str(x) for x in rhythm_out]),
#}

#json.dump(data, sys.stdout)