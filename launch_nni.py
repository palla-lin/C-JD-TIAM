# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/05 
"""
import nni
from nni.experiment import Experiment

search_space = {
    'emb_dim': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.001,0.00001]},
    'droput_rate': {'_type': 'uniform', '_value': [0.0, 0.5]},
    "batch_size": {"_type": "choice", "_value": [50, 250, 500]},
    'kernel_wins':{"_type": "choice", "_value": [[2,3,4],[1,2,3],[3,4,5]]},
    'dim_channel':{'_type': 'choice', '_value': [50, 100, 250, 500]},
}
command = 'CUDA_VISIBLE_DEVICES=3 python code/train_nni.py ' \
          '-bs 500 ' \
          '-ds 0 ' \
          '-output_dir output ' \
          '-num_layers 12 ' \
          '-wandb disabled'

experiment = Experiment('local')
experiment.config.trial_command = command
experiment.config.trial_code_directory = '.'
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.search_space = search_space
experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 2
experiment.run(18888)

input('Press enter to quit')
experiment.stop()