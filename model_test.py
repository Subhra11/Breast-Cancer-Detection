#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import keras


# In[ ]:


from tensorflow.keras.models import load_model
res_mod_org = load_model('/fred/oz138/COS80028/P5/Subhra/Breast_cancer/model-inception_resnet_v2-final.h5', compile=False)
res_mod_crp = load_model('/fred/oz138/COS80028/P5/Subhra/Breast_cancer/model-inception_resnet_v2-final_cropped.h5', compile=False)


# In[ ]:


from dm_image import DMImageDataGenerator
test_imgen = DMImageDataGenerator(featurewise_center=True)
test_imgen.mean = 52.18
test_generator = test_imgen.flow_from_directory(
    '/fred/oz138/COS80028/P5/Subhra/Breast_cancer/Data/INbreast_test', target_size=(1152, 896), target_scale=None,
    rescale_factor=0.003891,
    equalize_hist=False, dup_3_channels=True, 
    classes=['0', '1'], class_mode='categorical', batch_size=4, 
    shuffle=False)


# In[ ]:


from dm_keras_ext import DMAucModelCheckpoint
res_auc, res_y_true, res_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod_org, test_samples=test_generator.nb_sample, return_y_res=True)
print(res_auc)


# In[ ]:


from dm_keras_ext import DMAucModelCheckpoint
res_auc, res_y_true, res_y_pred = DMAucModelCheckpoint.calc_test_auc(
    test_generator, res_mod_crp, test_samples=test_generator.nb_sample, return_y_res=True)
print(res_auc)


# In[ ]:




