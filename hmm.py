import torch
import numpy as np

class HMM(torch.nn.Module):
  """
  Hidden Markov Model with discrete observations.
  """
  def __init__(self, M, N):
    super(HMM, self).__init__()
    self.M = M # number of possible observations
    self.N = N # number of states

    # A
    self.transition_model = TransitionModel(self.N)

    # b(x_t)
    self.emission_model = EmissionModel(self.N,self.M)

    # pi
    self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))

    # use the GPU
    self.is_cuda = torch.cuda.is_available()
    if self.is_cuda: self.cuda()

class TransitionModel(torch.nn.Module):
  def __init__(self, N):
    super(TransitionModel, self).__init__()
    self.N = N
    self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N,N))

class EmissionModel(torch.nn.Module):
  def __init__(self, N, M):
    super(EmissionModel, self).__init__()
    self.N = N
    self.M = M
    self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N,M))





def sample(self, T=10):
  state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
  transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
  emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

  # sample initial state
  z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
  z = []; x = []
  z.append(z_t)
  for t in range(0,T):
    # sample emission
    x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
    x.append(x_t)

    # sample transition
    z_t = torch.distributions.categorical.Categorical(transition_matrix[:,z_t]).sample().item()
    if t < T-1: z.append(z_t)

  return x, z

# Add the sampling method to our HMM class






import string

alphabet = string.ascii_lowercase
def encode(s):
  """
  Convert a string into a list of integers
  """
  x = [alphabet.find(ss) for ss in s]
  if x == -1:
     print('ERR!!!!!!!!!', s)
  return x

def decode(x):
  """
  Convert list of ints to string
  """
  s = "".join([alphabet[xx] for xx in x])
  return s

# # Initialize the model
# model = HMM(M=len(alphabet), N=2) 

# # Hard-wiring the parameters!
# # Let state 0 = consonant, state 1 = vowel
# for p in model.parameters():
#     p.requires_grad = False # needed to do lines below
# model.unnormalized_state_priors[0] = 0.    # Let's start with a consonant more frequently
# model.unnormalized_state_priors[1] = -0.5
# print("State priors:", torch.nn.functional.softmax(model.unnormalized_state_priors, dim=0))

# # In state 0, only allow consonants; in state 1, only allow vowels
# vowel_indices = torch.tensor([alphabet.index(letter) for letter in "aeiou"])
# consonant_indices = torch.tensor([alphabet.index(letter) for letter in "bcdfghjklmnpqrstvwxyz"])
# model.emission_model.unnormalized_emission_matrix[0, vowel_indices] = -np.inf
# model.emission_model.unnormalized_emission_matrix[1, consonant_indices] = -np.inf 
# print("Emission matrix:", torch.nn.functional.softmax(model.emission_model.unnormalized_emission_matrix, dim=1))

# # Only allow vowel -> consonant and consonant -> vowel
# model.transition_model.unnormalized_transition_matrix[0,0] = -np.inf  # consonant -> consonant
# model.transition_model.unnormalized_transition_matrix[0,1] = 0.       # vowel -> consonant
# model.transition_model.unnormalized_transition_matrix[1,0] = 0.       # consonant -> vowel
# model.transition_model.unnormalized_transition_matrix[1,1] = -np.inf  # vowel -> vowel
# print("Transition matrix:", torch.nn.functional.softmax(model.transition_model.unnormalized_transition_matrix, dim=0))











def HMM_forward(self, x, T):
    """
    x : IntTensor of shape (batch size, T_max)
    T : IntTensor of shape (batch size)

    Compute log p(x) for each example in the batch.
    T = length of each example
    """
    if self.is_cuda:
        x = x.cuda()
        T = T.cuda()
    batch_size = x.shape[0]; T_max = x.shape[1]
    log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
    log_alpha = torch.zeros(batch_size, T_max, self.N)
    if self.is_cuda: log_alpha = log_alpha.cuda()

    log_alpha[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
    for t in range(1, T_max):
        log_alpha[:, t, :] = self.emission_model(x[:,t]) + self.transition_model(log_alpha[:, t-1, :])

  # Select the sum for the final timestep (each x may have different length).
    log_sums = log_alpha.logsumexp(dim=2)
    log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)
    return log_probs

def emission_model_forward(self, x_t):
  log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
  out = log_emission_matrix[:, x_t].transpose(0,1)
  return out

def transition_model_forward(self, log_alpha):
  """
  log_alpha : Tensor of shape (batch size, N)
  Multiply previous timestep's alphas by transition matrix (in log domain)
  """
  log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

  # Matrix multiplication in the log domain
  out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
  return out

def log_domain_matmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Normally, a matrix multiplication
	computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	# log_A_expanded = torch.stack([log_A] * p, dim=2)
	# log_B_expanded = torch.stack([log_B] * m, dim=0)
    # fix for PyTorch > 1.5 by egaznep on Github:
	log_A_expanded = torch.reshape(log_A, (m,n,1))
	log_B_expanded = torch.reshape(log_B, (1,n,p))

	elementwise_sum = log_A_expanded + log_B_expanded
	out = torch.logsumexp(elementwise_sum, dim=1)

	return out





# x = torch.stack( [torch.tensor(encode("cat"))] )
# T = torch.tensor([3])
# print(model.forward(x, T))

# x = torch.stack( [torch.tensor(encode("aba")), torch.tensor(encode("abb"))] )
# T = torch.tensor([3,3])
# print(model.forward(x, T))





def viterbi(self, x, T):
  """
  x : IntTensor of shape (batch size, T_max)
  T : IntTensor of shape (batch size)
  Find argmax_z log p(x|z) for each (x) in the batch.
  """
  if self.is_cuda:
    x = x.cuda()
    T = T.cuda()

  batch_size = x.shape[0]; T_max = x.shape[1]
  log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
  log_delta = torch.zeros(batch_size, T_max, self.N).float()
  psi = torch.zeros(batch_size, T_max, self.N).long()
  if self.is_cuda:
    log_delta = log_delta.cuda()
    psi = psi.cuda()

  log_delta[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
  for t in range(1, T_max):
    max_val, argmax_val = self.transition_model.maxmul(log_delta[:, t-1, :])
    log_delta[:, t, :] = self.emission_model(x[:,t]) + max_val
    psi[:, t, :] = argmax_val

  # Get the log probability of the best path
  log_max = log_delta.max(dim=2)[0]
  best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

  # This next part is a bit tricky to parallelize across the batch,
  # so we will do it separately for each example.
  z_star = []
  for i in range(0, batch_size):
    z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
    for t in range(T[i] - 1, 0, -1):
      z_t = psi[i, t, z_star_i[0]].item()
      z_star_i.insert(0, z_t)

    z_star.append(z_star_i)

  return z_star, best_path_scores # return both the best path and its log probability

def transition_model_maxmul(self, log_alpha):
  log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

  out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0,1))
  return out1.transpose(0,1), out2.transpose(0,1)

def maxmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Similar to the log domain matrix multiplication,
	this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = torch.stack([log_A] * p, dim=2)
	log_B_expanded = torch.stack([log_B] * m, dim=0)

	elementwise_sum = log_A_expanded + log_B_expanded
	out1,out2 = torch.max(elementwise_sum, dim=1)

	return out1,out2





# x = torch.stack( [torch.tensor(encode("aba")), torch.tensor(encode("abb"))] )
# T = torch.tensor([3,3])
# print(model.viterbi(x, T))





import torch.utils.data
from collections import Counter
from sklearn.model_selection import train_test_split

class TextDataset(torch.utils.data.Dataset):
  def __init__(self, lines):
    self.lines = lines # list of strings
    collate = Collate() # function for generating a minibatch from strings
    self.loader = torch.utils.data.DataLoader(self, batch_size=1024, num_workers=1, shuffle=True, collate_fn=collate)

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx].lstrip(" ").rstrip("\n").rstrip(" ").rstrip("\n")
    return line

class Collate:
  def __init__(self):
    pass

  def __call__(self, batch):
    """
    Returns a minibatch of strings, padded to have the same length.
    """
    x = []
    batch_size = len(batch)
    for index in range(batch_size):
      x_ = batch[index].lower()

      # convert letters to integers
      x.append(encode(x_))

    # pad all sequences with 0 to have same length
    x_lengths = [len(x_) for x_ in x]
    T = max(x_lengths)
    for index in range(batch_size):
      x[index] += [0] * (T - len(x[index]))
      x[index] = torch.tensor(x[index])

    # stack into single tensor
    x = torch.stack(x)
    x_lengths = torch.tensor(x_lengths)
    return (x,x_lengths)
  






from tqdm import tqdm # for displaying progress bar

class Trainer:
  def __init__(self, model, lr):
    self.model = model
    self.lr = lr
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
  
  def train(self, dataset):
    train_loss = 0
    num_samples = 0
    self.model.train()
    print_interval = 50
    for idx, batch in enumerate(tqdm(dataset.loader)):
      x,T = batch
      batch_size = len(x)
      num_samples += batch_size
      log_probs = self.model(x,T)
      loss = -log_probs.mean()
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      train_loss += loss.cpu().data.numpy().item() * batch_size
      if idx % print_interval == 0:
        print("loss:", loss.item())
        for _ in range(5):
          sampled_x, sampled_z = self.model.sample()
          print(decode(sampled_x))
          print(sampled_z)
    train_loss /= num_samples
    return train_loss

  def test(self, dataset):
    test_loss = 0
    num_samples = 0
    self.model.eval()
    print_interval = 50
    for idx, batch in enumerate(dataset.loader):
      x,T = batch
      batch_size = len(x)
      num_samples += batch_size
      log_probs = self.model(x,T)
      loss = -log_probs.mean()
      test_loss += loss.cpu().data.numpy().item() * batch_size
      if idx % print_interval == 0:
        print("loss:", loss.item())
        sampled_x, sampled_z = self.model.sample()
        print(decode(sampled_x))
        print(sampled_z)
    test_loss /= num_samples
    return test_loss
  



if __name__ == '__main__':
    HMM.sample = sample
    alphabet = string.ascii_lowercase

    TransitionModel.forward = transition_model_forward
    EmissionModel.forward = emission_model_forward
    HMM.forward = HMM_forward

    TransitionModel.maxmul = transition_model_maxmul
    HMM.viterbi = viterbi

    filename = "training.txt"

    with open(filename, "r") as f:
        lines = f.readlines() # each line of lines will have one word

    alphabet = list(Counter(("".join(lines))).keys())
    train_lines, valid_lines = train_test_split(lines, test_size=0.1, random_state=42)
    train_dataset = TextDataset(train_lines)
    valid_dataset = TextDataset(valid_lines)

    M = len(alphabet)
    # Initialize model
    model = HMM(N=64, M=M)

    # Train the model
    num_epochs = 30
    trainer = Trainer(model, lr=0.01)

    for epoch in range(num_epochs):
            print("========= Epoch %d of %d =========" % (epoch+1, num_epochs))
            train_loss = trainer.train(train_dataset)
            valid_loss = trainer.test(valid_dataset)

            print("========= Results: epoch %d of %d =========" % (epoch+1, num_epochs))
            print("train loss: %.2f| valid loss: %.2f\n" % (train_loss, valid_loss) )