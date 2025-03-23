#!/usr/bin/env python3
"""
Dynamic Programming Solutions


This file contains implementations for a variety of classic dynamic programming problems:
- 0/1 Knapsack
- Longest Common Subsequence (LCS)
- Longest Palindromic Subsequence (LPS)
- Edit Distance
- Regular Expression Matching
- Rod Cutting
- Optimal Binary Search Tree
- Minimum Coin Change
- Matrix Chain Multiplication
- Subset Sum Problem
- Longest Increasing Subsequence (LIS)
- 2 Player Game (Optimal Strategy for a Game)
- Counting Paths in a Matrix
"""

# ----------------------------------
# 0/1 Knapsack
# ----------------------------------
def knapsack(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
    return dp[n][capacity]

# Example:
# values = [60, 100, 120]
# weights = [10, 20, 30]
# capacity = 50  => Output: 220

# ----------------------------------
# Longest Common Subsequence (LCS)
# ----------------------------------
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# Example:
# s1 = "abcde", s2 = "ace"  => Output: 3

# ----------------------------------
# Longest Palindromic Subsequence (LPS)
# ----------------------------------
def lps(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = 1
        
    # Build the table. Note: We fill dp table diagonally.
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + (dp[i + 1][j - 1] if length > 2 else 0)
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]

# Example:
# s = "bbbab"  => Output: 4

# ----------------------------------
# Edit Distance
# ----------------------------------
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],     # Deletion
                                   dp[i][j - 1],     # Insertion
                                   dp[i - 1][j - 1]) # Replacement
    return dp[m][n]

# Example:
# word1 = "horse", word2 = "ros"  => Output: 3

# ----------------------------------
# Regular Expression Matching
# Supports '.' and '*' in the pattern.
# ----------------------------------
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Deals with patterns like a*, a*b*, a*b*c* etc.
    for j in range(2, n + 1):
        if p[j - 1] == '*' and dp[0][j - 2]:
            dp[0][j] = True

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                # Zero occurrence of p[j-2]
                dp[i][j] = dp[i][j - 2]
                # One or more occurrence if matching the preceding element
                if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            else:
                dp[i][j] = False
    return dp[m][n]

# Example:
# s = "aab", p = "c*a*b"  => Output: True

# ----------------------------------
# Rod Cutting
# ----------------------------------
def rod_cutting(prices, n):
    # prices: list where prices[i] is the price of a rod of length i+1
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i):
            dp[i] = max(dp[i], prices[j] + dp[i - j - 1])
    return dp[n]

# Example:
# prices = [1, 5, 8, 9, 10, 17, 17, 20] for rod length 8  => Output: 22

# ----------------------------------
# Optimal Binary Search Tree (OBST)
# ----------------------------------
def optimal_bst(freq):
    # freq: list of frequencies for keys sorted in increasing order.
    n = len(freq)
    dp = [[0] * n for _ in range(n)]
    # prefix sum for efficient sum calculation
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]
    
    for i in range(n):
        dp[i][i] = freq[i]
        
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = float('inf')
            # sum of frequencies from i to j
            s = prefix[j + 1] - prefix[i]
            for k in range(i, j + 1):
                left = dp[i][k - 1] if k > i else 0
                right = dp[k + 1][j] if k < j else 0
                dp[i][j] = min(dp[i][j], left + right + s)
    return dp[0][n - 1]

# Example:
# freq = [34, 8, 50]  => Output: Optimal cost (varies based on frequencies)

# ----------------------------------
# Minimum Coin Change
# ----------------------------------
def min_coins(coins, total):
    dp = [float('inf')] * (total + 1)
    dp[0] = 0
    for coin in coins:
        for j in range(coin, total + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    return dp[total] if dp[total] != float('inf') else -1

# Example:
# coins = [1, 2, 5], total = 11  => Output: 3

# ----------------------------------
# Matrix Chain Multiplication
# ----------------------------------
def matrix_chain_order(dimensions):
    # dimensions: list of matrix dimensions such that the i-th matrix is dimensions[i-1] x dimensions[i]
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    
    # L is chain length.
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1]

# Example:
# dimensions = [40, 20, 30, 10, 30]
# Expected Output: 26000

# ----------------------------------
# Subset Sum Problem
# ----------------------------------
def subset_sum(nums, total):
    n = len(nums)
    dp = [[False] * (total + 1) for _ in range(n + 1)]
    
    # Sum 0 is always possible (empty subset)
    for i in range(n + 1):
        dp[i][0] = True
        
    for i in range(1, n + 1):
        for j in range(1, total + 1):
            if nums[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[n][total]

# Example:
# nums = [3, 34, 4, 12, 5, 2], total = 9  => Output: True

# ----------------------------------
# Longest Increasing Subsequence (LIS)
# ----------------------------------
def lis(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Example:
# nums = [10,9,2,5,3,7,101,18]  => Output: 4

# ----------------------------------
# 2 Player Game (Optimal Strategy for a Game)
# ----------------------------------
def optimal_game(coins):
    n = len(coins)
    dp = [[0] * n for _ in range(n)]
    # Base case: when there is only one coin.
    for i in range(n):
        dp[i][i] = coins[i]
    
    # Fill dp table for intervals of increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # When player picks the i-th coin:
            a = dp[i + 2][j] if i + 2 <= j else 0
            b = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
            # When player picks the j-th coin:
            c = dp[i][j - 2] if i <= j - 2 else 0
            dp[i][j] = max(coins[i] + min(a, b), coins[j] + min(b, c))
    return dp[0][n - 1]

# Example:
# coins = [8, 15, 3, 7]  => Output: 22

# ----------------------------------
# Counting Paths in a Matrix
# ----------------------------------
def count_paths(n, m):
    dp = [[0] * m for _ in range(n)]
    
    # Initialize first row and first column
    for i in range(n):
        dp[i][0] = 1
    for j in range(m):
        dp[0][j] = 1
        
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[n - 1][m - 1]

# Example:
# n = 3, m = 3  => Output: 6

# ----------------------------------
# Main function to test all DP solutions
# ----------------------------------
if __name__ == '__main__':
    print("0/1 Knapsack:", knapsack([60, 100, 120], [10, 20, 30], 50))  # Expected: 220

    print("LCS:", lcs("abcde", "ace"))  # Expected: 3

    print("LPS:", lps("bbbab"))  # Expected: 4

    print("Edit Distance:", edit_distance("horse", "ros"))  # Expected: 3

    print("Regex Matching:", is_match("aab", "c*a*b"))  # Expected: True

    print("Rod Cutting:", rod_cutting([1, 5, 8, 9, 10, 17, 17, 20], 8))  # Expected: 22

    print("Optimal BST:", optimal_bst([34, 8, 50]))  # Cost varies based on frequency

    print("Minimum Coin Change:", min_coins([1, 2, 5], 11))  # Expected: 3

    print("Matrix Chain Multiplication:", matrix_chain_order([40, 20, 30, 10, 30]))  # Expected: 26000

    print("Subset Sum:", subset_sum([3, 34, 4, 12, 5, 2], 9))  # Expected: True

    print("LIS:", lis([10, 9, 2, 5, 3, 7, 101, 18]))  # Expected: 4

    print("2 Player Game:", optimal_game([8, 15, 3, 7]))  # Expected: 22

    print("Counting Paths:", count_paths(3, 3))  # Expected: 6
