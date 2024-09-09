
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_space_separated_file(fileName):
    with open(fileName, "r") as f:
        # Reading from a file
        output = f.read()
    return np.array([[float(x) for x in s.split(' ') if x != ''] for s in output.split('\n') if s != ''])

fig, ax = plt.subplots()
A = read_space_separated_file("score.out")
pos = ax.imshow(A, interpolation='nearest', cmap='RdBu', vmin=-1, vmax=1)
fig.colorbar(pos, ax=ax)
plt.savefig("img/comp")

fig, ax = plt.subplots()
A = read_space_separated_file("first.out")
N, _ = A.shape
for i in range(9):
    ax.plot(range(N), A[:,i], label=f"Action {i}")
plt.legend()
plt.savefig("img/first")