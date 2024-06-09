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

data_input = json.loads(input())
bars = len(data_input['chords'])


#FLAT_NOTES = list(chain.from_iterable(measures))

rhythm = []
for j in range(bars):
    choice = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    for i in range(4):
        removal_index = random.randint(1, 7 - i)
        del choice[removal_index]
    rhythm_row = []
    for i in range(4):
        rhythm_row.append(choice[i + 1] - choice[i])
    rhythm.append(rhythm_row)

rhythm_out = list(chain.from_iterable(rhythm))

data = {
    'rhythm': list(chain.from_iterable(rhythm))
}

json.dump(data, sys.stdout)

#print(rhythm_out)

#data = {
#    'rhythm': ' '.join([str(x) for x in rhythm_out]),
#}

#json.dump(data, sys.stdout)