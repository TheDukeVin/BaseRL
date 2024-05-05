
import random

def sampleTime():
    while True:
        x = random.randint(1, 3)
        y = random.randint(1, 34)
        if (x,y) not in [(1, 34), (2, 2), (3, 34)]:
            return abs(x - 2) + abs(y - 2)

def sampleScore():
    t = 0
    s = 0
    while True:
        t += sampleTime()
        if t <= 100:
            s += 1
        else:
            return s

s = 0
for i in range(0, 1000000):
    s += sampleScore()
print(s)