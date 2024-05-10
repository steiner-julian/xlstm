import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

from xLSTM import mLSTM

def generate_sine_wave(seq_len, num_sequences):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    return torch.tensor(y).float().view(-1, 1).repeat(1, num_sequences).unsqueeze(0)

input_size = 1
hidden_size = 10
mem_dim = 10
seq_len = 100
num_sequences = 1

model = mLSTM(input_size, hidden_size, mem_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

data = generate_sine_wave(seq_len, num_sequences)

for epoch in range(200):
    states = model.init_hidden()
    optimizer.zero_grad()
    loss = 0
    for t in range(seq_len - 1):
        x = data[:, t]
        y_true = data[:, t + 1]
        y_pred, states = model(x, states)
        loss += criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')

test_output = []
states = model.init_hidden()
for t in range(seq_len - 1):
    x = data[:, t]
    y_pred, states = model(x, states)
    test_output.append(y_pred.detach().numpy().ravel()[0])

plt.figure(figsize=(10, 4))
plt.title('Original vs. Predicted Sine Wave')
plt.plot(data.numpy().ravel(), label='Original')
plt.plot(test_output, label='Predicted')
plt.legend()
plt.show()
