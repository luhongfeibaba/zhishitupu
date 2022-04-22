#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from tensorflow.python import pywrap_tensorflow
from Parameters import Parameters as pm
model_checkpoint_path = os.path.join(pm.model_checkpoint_path, 'best_validation3080-3080')
reader = pywrap_tensorflow.NewCheckpointReader(model_checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key, end='\n',)
    if key == 'global_step':
        print(reader.get_tensor(key))
    # print(reader.get_tensor(key))