
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 100
d = 3

X = np.random.normal(size=(d, N))
A = np.array([
    [3, 5, 1],
    [2, -1, 4],
    [1, 9, -4]
])
Y = (A @ X)
s = ""
for i in range(N):
    for j in range(d):
        s += str(Y[j,i])
        s += ' '
with open("data.out", "w") as f:
    f.write(s)