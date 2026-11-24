i = 0
total = 0
while i < 100000:
    values = [i, i + 1, i + 2]
    total = total + values[1]
    i = i + 1
print(total)
