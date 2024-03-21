import numpy as np

# assume this is your list
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# sort the list
lst_sorted = np.sort(lst)

# find the first quartile (q1)
q1 = np.percentile(lst_sorted, 80)

# find the indices of the values in the first quartile
# q1_indices = np.where(lst_sorted >= q1)

# find the median of the first quartile
print(q1)
q1_median = np.median(q1)

print(q1_median)