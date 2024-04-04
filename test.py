
s = 0.0
for i in range(100):
    for j in range(100):
        if i != j:
            s += abs(i%10 - j%10) + abs(i//10 - j//10)
print (s / (100*99))