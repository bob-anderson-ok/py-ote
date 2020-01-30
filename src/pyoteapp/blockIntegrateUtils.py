import numpy as np


# Return the standard deviation of a single integration block
# that starts at 'index'.   'index' is assumed to point to a value in y[], so
# we also check for a 'block' that goes past the end of y[]
def std_of_block(index, blockSize, y):
    if index + blockSize > len(y):
        return False, 0
    else:
        return True, np.std(y[index:(index+blockSize)])


# Return the mean of the standard deviations of all integration blocks in
# y[], starting at 'index' (assumed in 0...blockSize-1)
def mean_stds(index, blockSize, y):
    ans = 0
    n = 0
    i = index
    while True:
        ok, std = std_of_block(i, blockSize, y)
        if not ok:
            break
        else:
            ans += std
            n += 1
        i = i + blockSize

    if n > 0:
        return ans / n
    else:
        return ans


# Returns a list of the block standard deviations for all possible
# 'index' (offset) values
#
# np.argmin(ans) will get the 'best' offset value while
# np.min(ans)    will get the corresponding 'best' standard deviation
def mean_std_versus_offset(blockSize, y):
    ans = []
    for i in range(0, blockSize):  # i is the offset of the first block
        ans.append(mean_stds(i, blockSize, y))
    return ans
