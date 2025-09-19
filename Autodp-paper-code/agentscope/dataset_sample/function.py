import numpy as np
import pdb
import torch


class SubmodularFunction(object):
    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None


        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


class GraphCut(SubmodularFunction):
    def __init__(self, lam: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

        if 'similarity_matrix' in kwargs:
            self.sim_matrix_cols_sum = torch.sum(self.similarity_matrix, axis=0)
        self.all_idx = torch.ones(self.n, dtype=bool, device=torch.device('cuda'))

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = torch.zeros([self.n, self.n], dtype=torch.float32, device=torch.device('cuda'))
        self.sim_matrix_cols_sum = torch.zeros(self.n, dtype=torch.float32, device=torch.device('cuda'))
        self.if_columns_calculated = torch.zeros(self.n, dtype=bool, device=torch.device('cuda'))

        def _func(a, b):
            if not torch.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.sim_matrix_cols_sum[not_calculated] = torch.sum(self.sim_matrix[:, not_calculated], axis=0)
                self.if_columns_calculated[not_calculated] = True
            ##########复现self.sim_matrix[np.ix_(a, b)]
            a_indices = torch.nonzero(a).squeeze(1)
            b_indices = torch.nonzero(b).squeeze(1)
            return self.sim_matrix[a_indices][:, b_indices]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):

        # gain = -2. * torch.sum(self.similarity_kernel(selected, idx_gain), axis=0) + self.lam * self.sim_matrix_cols_sum[idx_gain]

        gain = self.lam * self.sim_matrix_cols_sum[idx_gain]

        return gain

    def update_state(self, new_selection, total_selected, **kwargs):
        pass
