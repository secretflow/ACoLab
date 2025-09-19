import numpy as np
from tqdm import tqdm
import torch
import pdb


optimizer_choices = ["NaiveGreedy", "LazyGreedy", "StochasticGreedy", "ApproximateLazyGreedy"]

class optimizer(object):
    def __init__(self, args, index, budget:int, already_selected=[]):
        self.args = args
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected


class NaiveGreedy(optimizer):
    def __init__(self, args, index, budget:int, already_selected=[]):
        super(NaiveGreedy, self).__init__(args, index, budget, already_selected)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = torch.zeros(self.n, dtype=bool, device=torch.device('cuda'))
        selected[self.already_selected] = True

        greedy_gain = torch.zeros(len(self.index), device=torch.device('cuda'))

        for i in tqdm(range(sum(selected), self.budget)):
            greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
            current_selection = greedy_gain.argmax()
            selected[current_selection] = True
            greedy_gain[current_selection] = -np.inf
            if update_state is not None:
                update_state(current_selection, selected, **kwargs)
        return self.index[selected]