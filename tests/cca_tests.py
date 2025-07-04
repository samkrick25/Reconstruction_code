import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([5,4,3,2,1])
z = np.stack((x,y))
def cov(x, y):
    if len(x) != len(y):
        print("lengths of x and y need to be equal!")
        return
    else:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        n = len(x)
        sum_of_products = np.sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(n)])
    
    return sum_of_products / (n - 1)
    
bool_val = cov(x, y) == np.cov(z)
print(bool_val, cov(x, y), np.cov(z))
