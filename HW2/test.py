import numpy as np

x = np.ones((3, 4))

y = np.random.rand(x.shape[0] * x.shape[1]).reshape(x.shape)
yp = np.random.rand(x.shape[0], x.shape[1])

print(y)
print(yp)

# print(np.random.rand(10))
