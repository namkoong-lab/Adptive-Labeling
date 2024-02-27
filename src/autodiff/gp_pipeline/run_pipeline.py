from typing import Callable, NamedTuple
import numpy as np
import pandas as pd
import torch
import dataclasses


import warnings
warnings.filterwarnings('ignore')

import gp_pipeline_regression
from constant_network import ConstantValueNetwork

url = '/shared/share_mala/yuanzhe/adaptive_sampling/pipeline_datasets/'
train_csv_name = url + 'train_csv_name'
test_csv_name = url + 'test_csv_name'
pool_csv_name = url + 'pool_csv_name'


df_train = pd.read_csv(train_csv_name)
print('training data count',df_train.shape)


dataset_cfg = gp_pipeline_regression.DatasetConfig(train_csv_name, test_csv_name, pool_csv_name, "EVENT_LABEL")
model_cfg = gp_pipeline_regression.ModelConfig(access_to_true_pool_y = True, hyperparameter_tune = False, batch_size_query = 2, temp_k_subset = 0.1, hidden_sizes_weight_NN = [50,50], meta_opt_lr = 0.001, meta_opt_weight_decay = 0.001)
train_cfg = gp_pipeline_regression.TrainConfig(n_train_iter = 100, N_iter = 100) #temp_var_recall is the new variable added here
gp_cfg = gp_pipeline_regression.GPConfig(length_scale=25.0, output_scale= 0.6931471824645996, noise = 1e-4, parameter_tune_lr = 0.001, parameter_tune_weight)decay = 0.001, paramater_tune_epochs = 100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_predictor = ConstantValueNetwork(constant_value=0.0, output_size=1).to(device)
#model_predictor = torch.jit.load('/user/dm3766/predictor_27_1.pt').to(device)

model_predictor.eval()

# Example usage
#need to add ``pipeline.'' in the below command; I didn't add it so that I can debug more easily
gp_pipeline_regression.experiment(dataset_cfg, model_cfg, train_cfg, enn_cfg, model_predictor, device, if_print = 1)

#%load_ext line_profiler
#profiling code - check which part
#%reload_ext line_profiler

#%lprun -f experiment experiment(dataset_cfg, model_cfg, train_cfg, enn_cfg, model_predictor, if_print = 1)

