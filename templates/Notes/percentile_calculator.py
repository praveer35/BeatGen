import math
import numpy as np
import random
import time


arr_N = 10000
loop_N = 1000
arr = [random.random() for _ in range(arr_N)]

start = time.time()
for _ in range(loop_N):
    a = 100 / (1 + math.exp(np.mean([math.log1p(-x) - math.log1p(x-1) for x in arr])))
end = time.time()

print(a, end - start)


start = time.time()
for _ in range(loop_N):
    a = 100 / (1 + math.pow(np.prod([(1-x) / x for x in arr]), 1/len(arr)))
end = time.time()

print(a, end - start)



start = time.time()
for _ in range(loop_N):
    a = 100 / (1 + np.prod([math.pow((1-x) / x, 1/len(arr)) for x in arr]))
end = time.time()

print(a, end - start)