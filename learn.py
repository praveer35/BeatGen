import numpy as np
from hmmlearn import hmm

# Define the note sequences for training
note_sequences = [
    ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    ['C', 'E', 'G', 'C', 'E', 'G', 'C'],
    ['G', 'F', 'E', 'D', 'C', 'D', 'E'],
    ['A', 'B', 'C', 'A', 'B', 'C', 'A']
]

# Convert note sequences to numerical representations
note_to_int = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
int_to_note = {i: note for note, i in note_to_int.items()}

X = np.array([[note_to_int[note] for note in sequence] for sequence in note_sequences])

# Define the HMM parameters
n_components = 7  # Number of hidden states
n_iter = 100  # Number of iterations for training

# Initialize the HMM model
model = hmm.CategoricalHMM(n_components=n_components, n_iter=n_iter)

# Train the HMM model
model.fit(X)

# Generate a new note sequence
generated_sequence_length = 10
generated_sequence = model.sample(generated_sequence_length)[0]

# Convert the generated sequence back to notes
generated_notes = [int_to_note[i[0]] for i in generated_sequence]

print("Generated Note Sequence:")
print(generated_notes)
print(model.transmat_)

"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

base_dir = "https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True"

data = pd.read_csv(base_dir)

# Convert the datetime from str to datetime object.
data["datetime"] = pd.to_datetime(data["datetime"])

# Determine the daily change in gold price.
data["gold_price_change"] = data["gold_price_usd"].diff()

# Restrict the data to later than 2008 Jan 01.
data = data[data["datetime"] >= pd.to_datetime("2008-01-01")]

# Plot the daily gold prices as well as the daily change.
plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
plt.plot(data["datetime"], data["gold_price_usd"])
plt.xlabel("datetime")
plt.ylabel("gold price (usd)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(data["datetime"], data["gold_price_change"])
plt.xlabel("datetime")
plt.ylabel("gold price change (usd)")
plt.grid(True)
plt.show()

# Use the daily change in gold price as the observed measurements X.
X = data[["gold_price_change"]].values
# Build the HMM model and fit to the gold price change data.
model = hmm.GaussianHMM(n_components = 3, covariance_type = "diag", n_iter = 50, random_state = 42)
model.fit(X)
# Predict the hidden states corresponding to observed X.
Z = model.predict(X)
states = pd.unique(Z)

plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_usd"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price (usd)", fontsize=16)
plt.subplot(2,1,2)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_change"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price change (usd)", fontsize=16)
plt.show()"""