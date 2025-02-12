import numpy as np

def dtw(melody1, melody2, w1=1, w2=1, w3=2, missing_penalty=10):
    n, m = len(melody1), len(melody2)
    D = np.full((n+1, m+1), float('inf'))
    D[0][0] = 0

    # Initialize edges (penalty for missing notes)
    for i in range(1, n+1):
        D[i][0] = D[i-1][0] + missing_penalty
    for j in range(1, m+1):
        D[0][j] = D[0][j-1] + missing_penalty

    # Distance metric
    def dist(a, b):
        if b == None:
            return w2 * a[1] + w3 * abs(a[2])
        return w1 * abs(a[0] - b[0]) + w2 * abs(a[1] - b[1]) + w3 * abs(a[2] - b[2])

    # Fill DP table
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(melody1[i-1], melody2[j-1])
            D[i][j] = min(
                D[i-1][j] + missing_penalty,   # Skip melody1[i-1]
                D[i][j-1] + missing_penalty,   # Skip melody2[j-1]
                D[i-1][j-1] + cost             # Align melody1[i-1] with melody2[j-1]
            )
    
    # DTW cost
    dtw_cost = D[n][m]
    
    # Maximum possible cost
    max_cost = (n + m) * missing_penalty
    
    # Normalized difference
    normalized_difference = (dtw_cost / max_cost) * 100

    return normalized_difference


M1 = [[0,4,-3], [4,4,-5], [8,2,-7], [10,4,-6], [14,2,-5]]
M2 = [[0,4,-3], [4,6,-10], [10,4,-6], [14,2,-5]]

difference = dtw(M1, M2)
print("Difference:", difference)