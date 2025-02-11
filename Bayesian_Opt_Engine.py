import random
import math
import numpy as np
import json
from skopt import gp_minimize
# import os
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
# import matplotlib.pyplot as plt


# chords = [6, 1, 2, 3]
# melody = [[7, 0.5], [4, 0.5], [2, 0.5],
#           [1, 1.0], [2, 0.5], [3, 0.5],
#           [2, 0.5], [4, 1.0], [3, 1.0],
#           [4, 0.5], [2, 1.0], [4, 0.5],
#           [7, 0.5], [4, 0.5], [2, 0.5],
#           [1, 1.0], [2, 0.5], [3, 0.5],
#           [2, 0.5], [4, 1.0], [3, 1.0],
#           [4, 0.5], [2, 1.0], [4, 0.5]]

class Bayesian_Opt_Engine:
    def __init__(self, chords=None, melody=None, generations=None):
        self.chords = chords
        self.bars = len(chords)
        self.melody = melody
        self.generations = generations
        self.max_input = None
        self.max_geometric_mean = 0
        self.geometric_mean_history = []
        self.max_geometric_mean_history = []
        self.max_confidence_index = 0
        self.confidence_index_history = []
        self.max_confidence_index_history = []

    def engine(self, x):
        global bayesian_probabilities
        bayesian_probabilities = []



        pitch_viscosity = int(x[0]*20)
        hook_chord_boost_onchord = x[1]*20
        hook_chord_boost_2_and_6 = x[2]*5
        hook_chord_boost_7 = x[3]*5
        hook_chord_boost_else = x[4]*5
        nonhook_chord_boost_onchord = x[5]*5
        nonhook_chord_boost_2_and_6 = x[6]*5
        nonhook_chord_boost_7 = x[7]*5
        nonhook_chord_boost_else = x[8]*5
        already_played_boost = x[9]*10


        rhythm = []
        measure_melody = []
        temp_rhythm = []
        temp_melody = []
        measure_sum = 0
        for i in range(len(self.melody)):
            if measure_sum < 4:
                temp_rhythm.append(self.melody[i][1])
                temp_melody.append(self.melody[i][0])
                measure_sum += self.melody[i][1]
            else:
                rhythm.append(temp_rhythm)
                measure_melody.append(temp_melody)
                temp_rhythm = [self.melody[i][1]]
                temp_melody = [self.melody[i][0]]
                measure_sum = self.melody[i][1]
        if len(temp_rhythm) > 0:
            rhythm.append(temp_rhythm)
            measure_melody.append(temp_melody)


        keynote_expectations = [measure_melody[0][0] - 2]
        for i in range(len(measure_melody)-1):
            keynote_expectations.append(measure_melody[i+1][0] - measure_melody[i][0])

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

            keynotes = [0] * self.bars

            for i in range(int(self.bars / 1)):
                markov_vector = recalculate_markov_vector(last_note, self.chords[i])
                P = markov_vector[7 + keynote_expectations[i]]
                index = 7 + keynote_expectations[i]
                global bayesian_probabilities
                bayesian_probabilities.append(P)
                last_note += (index - 7)
                keynotes[i] = last_note

            keynotes.append(keynotes[0])

            note = 0

            for i in range(int(self.bars / 1)):
                if i > 1 and measure_melody[i] == measure_melody[i-2]:
                    continue
                last_note = keynotes[i]
                temp_notes = []
                temp_notes.append(last_note)
                notes_played.add(last_note)
                measure_flutter = len(rhythm[i])
                for j in range(measure_flutter-1):
                    delta = round((keynotes[i+1] - last_note) / (measure_flutter - j - 1))
                    markov_vector2 = recalculate_markov_vector2(last_note, self.chords[i], delta, rhythm[i][j+1])
                    if len(markov_vector2) == 1:
                        bayesian_probabilities = []
                        return loop()
                    
                    expected_delta = measure_melody[i][j+1] - last_note
                    P = markov_vector2[7 + expected_delta]
                    index = 7 + expected_delta
                    bayesian_probabilities.append(P)
                    last_note += (index - 7)
                    temp_notes.append(last_note)
                    notes_played.add(last_note)
                measures.append(temp_notes)
                note += 1
            
            geometric_mean = math.pow(np.prod(bayesian_probabilities), 1/len(bayesian_probabilities))
            confidence_index = 0 if 0 in bayesian_probabilities else 100 / (1 + math.pow(np.prod([(1-x) / x for x in bayesian_probabilities]), 1/len(bayesian_probabilities)))

            return -geometric_mean, -confidence_index

        return loop()
    
    # def boom(self, x):
    #     try:
    #         y = self.engine(x)
    #         return y[1]
    #     except:
    #         return 0
    
    def get_likelihood(self, x):
        # y = None
        # for _ in range(self.n):
        #     y = self.boom(x)
        # print(y)
        # return y
        try:
            y = self.engine(x)
            print(f"{str(y):<{45}} {str(x)}")
            return y[1]
        except Exception as e:
            print(f"{"(0.0, 0.0)":<{45}} {str(x)}")
            return 0
        
    def get_batch_likelihood(self, x):
        sum = 0
        for generation in self.generations:
            self.chords = generation['chords']
            self.melody = generation['melody']
            sum += self.get_likelihood(x)
        return sum / len(self.generations)
    
    def naive_optimization(self, n):
        for _ in range(n):
            x = [random.random() for _ in range(10)]
            # x = [int(random.random()*20), random.random()*20, random.random()*5,
            #      random.random()*5, random.random()*5, random.random()*5, random.random()*5,
            #      random.random()*5, random.random()*5, random.random() * 10]
            geometric_mean, confidence_index = self.get_likelihood(x)
            self.max_geometric_mean = max(self.max_geometric_mean, geometric_mean)
            self.geometric_mean_history.append(geometric_mean*100)
            self.max_geometric_mean_history.append(self.max_geometric_mean*100)
            self.max_confidence_index = max(self.max_confidence_index, confidence_index)
            self.confidence_index_history.append(confidence_index)
            self.max_confidence_index_history.append(self.max_confidence_index)
            self.max_input = [i for i in x] if self.max_confidence_index == confidence_index else self.max_input

    def bayesian_optimization(self, n):
        res = gp_minimize(self.get_likelihood if self.melody else self.get_batch_likelihood, # the function to minimize
                        [(0.0, 1.0) for _ in range(10)],      # the bounds on each dimension of x
                        acq_func="EI",      # the acquisition function
                        n_calls=n,         # the number of evaluations of f
                        n_random_starts=5,  # the number of random initialization points
                        noise=0.1**2,       # the noise level (optional)
                        random_state=1234)   # the random seed
        
        # print(res.x)
        # print(res.fun)

        x = res.x
        # x = np.dot(x, [20, 20, 5, 5, 5, 5, 5, 5, 5, 10])
        x[0] = int(x[0]*20)
        x[1] = x[1]*20
        x[2] = x[2]*5
        x[3] = x[3]*5
        x[4] = x[4]*5
        x[5] = x[5]*5
        x[6] = x[6]*5
        x[7] = x[7]*5
        x[8] = x[8]*5
        x[9] = x[9]*10
        
        return x


# eng = Bayesian_Opt_Engine(chords, melody, 1)
# eng2 = Bayesian_Opt_Engine(chords, melody, 500)
# N = 75

# import time

# s = time.time()
# eng.bayesian_optimization(n=N)
# e = time.time()
# print("TIME:", e - s)

# s = time.time()
# eng2.bayesian_optimization(n=N)
# e = time.time()
# print("TIME:", e - s)

# eng.naive_optimization(n=N)

# print(eng.max_geometric_mean, eng.max_confidence_index, eng.max_input)

# plt.figure(facecolor='#003', figsize=(24,8))

# plt.plot([x for x in range(N)], eng.confidence_index_history, color='white', linewidth=3)
# plt.plot([x for x in range(N)], eng.max_confidence_index_history, color='lightgreen', linewidth=7)
# plt.plot([x for x in range(N)], eng.geometric_mean_history, color='purple', linewidth=3)
# plt.plot([x for x in range(N)], eng.max_geometric_mean_history, color='red', linewidth=7)
# plt.title(f"Convergence optimization (N={N})")

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