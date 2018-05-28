import pandas as pd
import os
from tensor_model import TensorModel

print (os.getcwd())



tensorModel = TensorModel(data)

tensorModel.initialize()
#tensorModel.fit_model()
arrData = data.values
for i in range(30):
    print('Computed: ', tensorModel.compute(arrData[i+100, :18]))
    print('Real: ', arrData[i+100, 18:20])