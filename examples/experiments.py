import os

BACKEND = 'torch'
os.environ['KERAS_BACKEND'] = BACKEND

import yaml
import itertools
import numpy as np
import random

import torch

from tkan import get_path
from pendulum_train_eval import train as train_pendulum
from pendulum_train_eval import test as test_pendulum

from mg_train_eval import create_data as create_data_mg
from mg_train_eval import train as train_mg
from mg_train_eval import test as test_mg


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


exps = "../../KANs"

with open(f'{exps}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

experiment = 'mackey_glass'

if experiment == 'pendulum':

    pendulum = config['pendulum']

    pendulum_length = pendulum['length']
    pendulum_seeds = pendulum['seeds']
    pendulum_lengths = pendulum['sequence_length']

    combinations = list(itertools.product(pendulum_length, pendulum_seeds, pendulum_lengths))

    for combination in combinations:
        dir_path = "../examples/experiments/pendulum"
        for c in combination:
            dir_path = f'{dir_path}/{c}'

        dir_path = get_path(dir_path)

        length, seed, sequence_length = combination[0], combination[1], combination[2]

        set_seeds(seed)

        print(f'Pendulum dataset with L={length}, seed={seed} and sequence length={sequence_length}')

        train_pendulum(dir_path, sequence_length)
        test_pendulum(dir_path, sequence_length)


elif experiment == 'mackey_glass':

    mg = config['mackey_glass']

    pendulum_seeds = mg['seeds']
    pendulum_lengths = mg['sequence_length']

    combinations = list(itertools.product(pendulum_seeds, pendulum_lengths))

    dir_path = "../examples/experiments/mackey_glass"
    dir_path = get_path(dir_path)
    data_train, data_val, data_test = create_data_mg(data_pth=f'{exps}/mackey_glass/data')

    for combination in combinations:
        dir_path = "../examples/experiments/mackey_glass"
        for c in combination:
            dir_path = f'{dir_path}/{c}'

        dir_path = get_path(dir_path)
        seed, sequence_length = combination[0], combination[1]
        set_seeds(seed)

        print(f'Mackey-Glass dataset with seed={seed} and sequence length={sequence_length}')

        train_mg(dir_path=dir_path, sequence_length=sequence_length, data_train=data_train, data_val=data_val)
        test_mg(dir_path=dir_path, sequence_length=sequence_length, data_test=data_test)

else:
    raise ValueError('Unknown experiment.')
