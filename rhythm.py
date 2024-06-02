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




#FLAT_NOTES = list(chain.from_iterable(measures))

rhythm = []
for j in range(4):
    choice = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    for i in range(4):
        removal_index = random.randint(1, 7 - i)
        del choice[removal_index]
    rhythm_row = []
    for i in range(4):
        rhythm_row.append(choice[i + 1] - choice[i])
    rhythm.append(rhythm_row)

rhythm_out = list(chain.from_iterable(rhythm))
print(' '.join([str(x) for x in rhythm_out]))
#print(rhythm_out)

#data = {
#    'rhythm': ' '.join([str(x) for x in rhythm_out]),
#}

#json.dump(data, sys.stdout)