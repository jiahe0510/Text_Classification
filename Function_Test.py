import Models
import numpy as np


model = Models.Models()
array = np.random.rand(5, 5)
print(model.binary_feature(array))