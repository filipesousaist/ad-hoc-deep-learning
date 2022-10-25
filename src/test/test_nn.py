import numpy as np
import time

from src.lib.models.NNModel import NNModel

t = time.time()
model = NNModel(1, 100)

#  **3
#  02*
#  1**
x = np.array([
    [-0.01, 0],
    [-0.01, 0.01],
    [0, 0],
    [0.01, -0.01]
]) * 100

X = x
for i in range(1, 1000):
    X = np.concatenate((X, x + 100 * i))

y = np.arange(len(X))

model.fit(X, y)

print(model.predict(np.array([[20.5, 21.5]])))

#print(model.index.search(np.array([[-100, 100]]).astype(np.float32), 5))
print(time.time() - t, "s")