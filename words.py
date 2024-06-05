import numpy as np
from hmmlearn import hmm
from collections import Counter

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

def create_char_mapping(words):
    char_counter = Counter(char for word in words for char in word)
    char_list = [char for char, _ in char_counter.most_common()]
    char_to_idx = {char: idx for idx, char in enumerate(char_list)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def words_to_sequences(words, char_to_idx):
    return [[char_to_idx[char] for char in word] for word in words]

"""common_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
]
with open('words.txt', 'w') as file:
    for word in common_words:
        file.write(f"{word}\n")"""
words = load_dataset('words.txt')
char_to_idx, idx_to_char = create_char_mapping(words)
sequences = words_to_sequences(words, char_to_idx)
lengths = [len(seq) for seq in sequences]
sequences_flat = [item for sublist in sequences for item in sublist]
sequences_flat = np.array(sequences_flat).reshape(-1, 1)

num_states = 20
model = hmm.GaussianHMM(n_components=num_states, n_iter=1000, tol=0.01)
model.fit(sequences_flat, lengths)

def generate_fake_words(model, idx_to_char, num_words=10, max_length=10):
    fake_words = []
    for _ in range(num_words):
        X, _ = model.sample(max_length)
        word = ''.join(idx_to_char[int(x)] for x in X.flatten())
        fake_words.append(word)
    return fake_words

fake_words = generate_fake_words(model, idx_to_char)
for word in fake_words:
    print(word)
