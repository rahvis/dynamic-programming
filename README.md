# Dynamic Programming Cheatsheet

## 0/1 Knapsack

### Input
```
value[], weight[], Capacity
```

### DP State
```
dp[i][j] represents max sum of value we get by using items from 0 to i and bag of capacity j
```
```
i = 0 to n
j = 0 to Capacity
```

### Transition
```
dp[i][j] =
  dp[i-1][j]      if j < weight[i]
  max(dp[i-1][j], value[i] + dp[i-1][j-weight[i]])    otherwise
```

---

## Longest Common Subsequence (LCS)

### Input
```
String s1 and String s2
```

### DP State
```
dp[i][j] represents LCS of s1.substring(0,i) and s2.substring(0,j)
```
```
dp[i][0] = dp[j][0] = 0
```
```
i = 1 to s1.length
j = 1 to s2.length
```

### Transition
```
dp[i][j] =
  dp[i-1][j-1]     if s1[i-1] == s2[j-1]
  max(dp[i-1][j], dp[i][j-1])     otherwise
```

---

## Longest Palindromic Subsequence (LPS)

### Input
```
String s
```

### DP State
```
dp[i][j] represents longest palindromic subsequence's length of substring(i, j), here i, j represent left, right indexes in the string
```
```
dp[i][i] = 1 // single character palindrome
```
```
Run matrix loop diagonally because dp[i][j] depends on dp[i+1][j-1]
len = 1 to s.length
i = 0 to s.length - len
j = i+len-1
```

### Transition
```
dp[i][j] =
  2 + dp[i+1][j-1]     if s[i] == s[j]
  max(dp[i+1][j], dp[i][j-1])    otherwise
```

---

## Edit Distance

### Input
```
String s1 and String s2
```

### DP State
```
dp[i][j] represents minimum number of operations to convert s1.substring(0,i) to s2.substring(0,j) using replace, insert, or delete operations.
```
```
dp[i][0] = i
dp[0][j] = j
```
```
i = 1 to s1.length
j = 1 to s2.length
```

### Transition
```
dp[i][j] =
  dp[i-1][j-1]   if s1[i] == s2[j]
  1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])  // min(replace, insert, delete)
```

---

## Regular Expression Matching

### Input
```
text String s and pattern String p
```

### DP State
```
dp[i][j] represents whether s.substring(0,i) matches with p.substring(0,j). It contains boolean value.
```
```
dp[0][0] = true
dp[0][j] = dp[0][j-2] if p[j-1] == '*'
```
```
i = 1 to s.length
j = 1 to p.length
```

### Transition
```
dp[i][j] =
  dp[i-1][j-1]     if s[i] == p[j] or p[j] == '.'
  dp[i][j-2] || (dp[i-1][j] && (s[i] == p[j-1] or p[j-1] == '.'))    if p[j] == '*'
  false   otherwise
```

---

## Rod Cutting

### Input
```
array price[] where price[i] contains price of rod of length i
```

### DP State
```
dp[i] represents max value we can get for rod of length i
```
```
i = 1 to n
```

### Transition
```
dp[i] =
 for j in 0 to i-1
   max(dp[i], price[j] + dp[i-j-1])
```

---

## Minimum Coin Change

### Input
```
array coins[] where coins[i] is denomination and total for which we need to find minimum coin change.
```

### DP State
```
dp[i][j] represents minimum coin change required to create value j using coins from 0 to i
```
```
i = 0 to coins.length
j = 0 to total
```

### Transition
```
dp[i][j] =
  min(1+dp[i][j - coins[i]], dp[i-1][j])   if j >= coins[i]
  dp[i-1][j]   otherwise
```

---

## Longest Increasing Subsequence (LIS)

### Input
```
array nums[] containing n integers
```

### DP State
```
dp[i] represents longest increasing subsequence till i index where nums[i] is the largest element or nums[i] included in increasing subsequence.
```

### Transition
```
dp[i] = 1
for j in 0 to i-1:
  dp[i] = max(dp[i], 1 + dp[j])   if nums[j] < nums[i]
```

---

## Counting Paths in a Matrix

### Input
```
n x m matrix, count number of ways to reach from top left to bottom right corner.
```

### DP State
```
dp[i][j] represents number of ways to reach from cell(0,0) to cell(i,j)
```
```
dp[0][0] = 1
```
```
i = 0 to n-1
j = 0 to m-1
```

### Transition
```
dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

