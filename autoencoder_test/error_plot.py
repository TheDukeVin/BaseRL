
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_space_separated_file(fileName):
    with open(fileName, "r") as f:
        # Reading from a file
        output = f.read()
    return np.array([[x for x in s.split(' ') if x != ''] for s in output.split('\n') if s != ''])

A = read_space_separated_file("auto_error.out").astype(float)
_, N = A.shape
plt.plot(np.arange(N), A[0, :])
plt.show()