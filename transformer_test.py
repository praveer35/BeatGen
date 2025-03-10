import random
import math

class RhythmGenerator:
    def generate_bar(self, f=0.0):
        if f < 0.001: f = 0.001
        if f > 0.999: f = 0.999
        f = math.sqrt(f)
        flutter = math.exp(math.log(f) - math.log(1-f))
        durations = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # Possible durations
        target_sum = 4.0  # Total bar length
        rhythm = []
        total_sum = 0.0

        # Generate a weight distribution for durations
        base_weights = {0.5: 0.5, 1.0: 1.2, 1.5: 1.0, 2.0: 0.8, 3.0: 0.5, 4.0: 1.3}

        #flutter = math.sqrt(flutter)
        
        # Adjust weights based on flutter: more flutter → favor shorter notes
        weights = {d: base_weights[d] * (1.5 - flutter + (flutter * (1.5 / d))) for d in durations}
        
        while total_sum < target_sum:
            # Filter out notes that would exceed the remaining sum
            available_durations = [d for d in durations if total_sum + d <= target_sum]
            if not available_durations:
                break
            
            # Normalize weights for available durations
            norm_weights = [weights[d] for d in available_durations]
            total_weight = sum(norm_weights)
            probabilities = [w / total_weight for w in norm_weights]

            # Choose a note duration based on skewed distribution
            note_duration = random.choices(available_durations, weights=probabilities)[0]
            rhythm.append(note_duration)
            total_sum += note_duration

        # Final adjustment to ensure the sum is exactly 4.0
        if total_sum < target_sum:
            rhythm.append(round(target_sum - total_sum, 1))

        return rhythm

    def engine(self, data_input, f=0.3):
        notes = data_input['notes']
        #notes = [-1]
        bars = len(notes)
        rhythm = []
        for _ in range(bars):
            rhythm.append(self.generate_bar(f))

        data = {
            #'rhythm': list(chain.from_iterable(final_rhythm)),
            'rhythm': rhythm
        }

        print(data)

        return data

# 🌍 Testing (Flutter dramatically changes the distribution)
# for f in [0.01, 0.5, 0.99]:
#     notes = 0
#     flutter = math.exp(math.log(f) - math.log(1-f))
#     print(flutter)
#     for _ in range(5000):
#         notes += len(generate_rhythm(flutter))
#         #print(f"Flutter {f}: {generate_rhythm(f)} (sum={sum(generate_rhythm(f))})")
#     print(notes/5000)
