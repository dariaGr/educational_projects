# import the necessary modules 
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import requests
import string
import textwrap
import os

### substitution cipher creation

# one for keys, another for values
letters_keys = list(string.ascii_lowercase)
letters_values = list(string.ascii_lowercase)

real_mapin_stateng = {}

# shuffle set of letters values
random.shuffle(letters_values)

# filling the map with data
for k, v in zip(letters_keys, letters_values):
  real_mapin_stateng[k] = v

### the language model

# initialize Markov matrix
M = np.ones((26, 26))

# initial state distribution
in_state = np.zeros(26)

# a function to update the Markov matrix
def update_transition(ch1, ch2):
  # change char to int
  i = ord(ch1) - 97
  j = ord(ch2) - 97
  M[i,j] += 1

# a function to update the initial state distribution
def update_in_state(ch):
  i = ord(ch) - 97
  in_state[i] += 1

# get the log-probability of a word / token
def get_word_prob(word):
  # print("word:", word)
  i = ord(word[0]) - 97
  logp = np.log(in_state[i])

  for ch in word[1:]:
    j = ord(ch) - 97
    logp += np.log(M[i, j]) # update prob
    i = j # update j

  return logp

# get the probability of a sequence of words
def get_sequence_prob(words):
  # if input is a string, split into an array of tokens
  if type(words) == str:
    words = words.split()

  logp = 0
  for word in words:
    logp += get_word_prob(word)
  return logp

### create a markov model based on an English dataset
# is an edit of http://www.gutenberg.org/ebooks/1400

# download the file
if not os.path.exists('great expectations.txt'):
  print("Downloading 'Great Expectations'...")
  r = requests.get('http://www.gutenberg.org/files/1400/1400-0.txt')
  with open('great expectations.txt', 'w', encoding="utf-8") as f:
    f.write(r.content.decode())

# for replacing non-alphabetic characters
regex = re.compile('[^a-zA-Z]')

# load in words
for line in open('great expectations.txt', encoding="utf-8"):
  line = line.rstrip()

  # there are blank lines in the file
  if line:
    line = regex.sub(' ', line) # replace all non-alphabetic characters with space

    # split the tokens in the line and lowercase
    tokens = line.lower().split()

    # update the model
    for token in tokens:

      # first letter
      ch0 = token[0]
      update_in_state(ch0)

      # other letters
      for ch1 in token[1:]:
        update_transition(ch0, ch1)
        ch0 = ch1

# normalize the probabilities
in_state /= in_state.sum()
M /= M.sum(axis=1, keepdims=True)

### encode a message

# this is a random excerpt from Project Gutenberg's
# Alice's Adventures in Wonderland by Lewis Carroll
# http://www.gutenberg.org/ebooks/11

original_message = '''‘Would it be of any use, now,’ thought Alice, ‘to speak to this mouse?
Everything is so out-of-the-way down here, that I should think very
likely it can talk: at any rate, there’s no harm in trying.’ So she
began: ‘O Mouse, do you know the way out of this pool? I am very tired
of swimming about here, O Mouse!’ (Alice thought this must be the right
way of speaking to a mouse: she had never done such a thing before, but
she remembered having seen in her brother’s Latin Grammar, ‘A mouse--of
a mouse--to a mouse--a mouse--O mouse!’) The Mouse looked at her rather
inquisitively, and seemed to her to wink with one of its little eyes,
but it said nothing. 
'''

# a function to encode a message
def encode_message(msg):
  # downcase
  msg = msg.lower()

  # replace non-alphabetic characters
  msg = regex.sub(' ', msg)

  # make the encoded message
  coded_msg = []
  for ch in msg:
    coded_ch = ch # could just be a space
    if ch in real_mapin_stateng:
      coded_ch = real_mapin_stateng[ch]
    coded_msg.append(coded_ch)

  return ''.join(coded_msg)


encoded_message = encode_message(original_message)


# a function to decode a message
def decode_message(msg, word_map):
  decoded_msg = []
  for ch in msg:
    decoded_ch = ch # could just be a space
    if ch in word_map:
      decoded_ch = word_map[ch]
    decoded_msg.append(decoded_ch)

  return ''.join(decoded_msg)

### run an evolutionary algorithm to decode the message

# this is our initialization point
dna_pool = []
for _ in range(20):
  dna = list(string.ascii_lowercase)
  random.shuffle(dna)
  dna_pool.append(dna)

def evolve_offspring(dna_pool, n_children):
  # make n_children per offspring
  offspring = []

  for dna in dna_pool:
    for _ in range(n_children):
      copy = dna.copy()
      j = np.random.randint(len(copy))
      k = np.random.randint(len(copy))

      # switch
      tmp = copy[j]
      copy[j] = copy[k]
      copy[k] = tmp
      offspring.append(copy)

  return offspring + dna_pool

num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')
for i in range(num_iters):
  if i > 0:
    # get offspring from the current dna pool
    dna_pool = evolve_offspring(dna_pool, 3)

  # calculate score for each dna
  dna2score = {}
  for dna in dna_pool:
    # populate map
    current_map = {}
    for k, v in zip(letters_keys, dna):
      current_map[k] = v

    decoded_message = decode_message(encoded_message, current_map)
    score = get_sequence_prob(decoded_message)

    # store it
    # needs to be a string to be a dict key
    dna2score[''.join(dna)] = score

    # record the best so far
    if score > best_score:
      best_dna = dna
      best_map = current_map
      best_score = score

  # average score for this generation
  scores[i] = np.mean(list(dna2score.values()))

  # keep the best 5 dna
  # also turn them back into list of single chars
  sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
  dna_pool = [list(k) for k, v in sorted_dna[:5]]

  if i % 200 == 0:
    print("iter:", i, "score:", scores[i], "best so far:", best_score)

# use best score
decoded_message = decode_message(encoded_message, best_map)

print("LL of decoded message:", get_sequence_prob(decoded_message))
print("LL of true message:", get_sequence_prob(regex.sub(' ', original_message.lower())))


# which letters are wrong?
for true, v in real_mapin_stateng.items():
  pred = best_map[v]
  if true != pred:
    print("true: %s, pred: %s" % (true, pred))

# print the final decoded message
print("Decoded message:\n", textwrap.fill(decoded_message))

print("\nTrue message:\n", original_message)

plt.plot(scores)
plt.show()
