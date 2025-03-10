import math

def stochastize(arr):
    sum = 0
    for num in arr:
        sum += num
    for i in range(len(arr)):
        arr[i] /= sum
    return arr

def skew_distribution(arr, c=0):
    # adjusted sigmoid function S, with constraints:
    # S(0) = 0
    # S(0.5) = 0.5
    # S(1) = 1
    # S''(x) > 0 for x in [0, 0.5)
    # S''(x) = 0 for x = 0.5
    # S''(x) < 0 for x in (0.5, 1]
    if c == 0: return arr
    return stochastize([0.5 + 0.5 * (2/(1-math.exp(-c/2)) - 1) * (2/(1+math.exp(c*(0.5-x))) - 1) for x in arr])


print(skew_distribution([0.2, 0.5, 0.2], c=10))