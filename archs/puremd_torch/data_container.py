import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp

index_keys = [
    "batch_seg",
    "idnb_i",
    "idnb_j",
    "id_expand_kj",
    "id_reduce_ji",
    "id3dnb_i",
    "id3dnb_j",
    "id3dnb_k",
]


class DataContainer(Dataset):
    def __init__(self, filename, cutoff, target_keys, device="cuda:0", seed=42, randomized=False, batch_size=32):
        data_dict = np.load(filename, allow_pickle=True)
        self.cutoff = cutoff
        self.target_keys = target_keys
        for key in ["id", "N", "Z", "R"]:
            if key in data_dict:
                setattr(self, key, data_dict[key])
            else:
                setattr(self, key, None)
        self.targets = np.stack([data_dict[key] for key in self.target_keys], axis=1)

        if self.N is None:
            self.N = np.zeros(len(self.targets), dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.device = device

        self._random_state = np.random.RandomState(seed=seed)
        all_idx = np.arange(self.targets.shape[0])
        if randomized:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        # slice the data indexes based on the batch size
        self.batch_sample = [all_idx[i:i+batch_size] for i in range(0, len(all_idx), batch_size)]

        # Store indices of training, validation and test data
        # self.idx = {'train': all_idx[0:ntrain],
        #             'val': all_idx[ntrain:ntrain+nvalid],
        #             'test': all_idx[ntrain+nvalid:]}

        assert self.R is not None

    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))]
        )

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))]
        )
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    def __len__(self):
        # return self.targets.shape[0]
        return len(self.batch_sample)

    def __getitem__(self,  ids):
        idx = self.batch_sample[ids]

        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {"targets": self.targets[idx], "id": self.id[idx], "N": self.N[idx]}
        data["batch_seg"] = np.repeat(np.arange(len(idx), dtype=np.int32), data["N"])
        adj_matrices = []

        data["Z"] = np.zeros(np.sum(data["N"]), dtype=np.int32)
        data["R"] = np.zeros([np.sum(data["N"]), 3], dtype=np.float32)

        nend = 0
        for k, i in enumerate(idx):
            n = data["N"][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data["Z"][nstart:nend] = self.Z[self.N_cumsum[i] : self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i] : self.N_cumsum[i + 1]]
            data["R"][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))
            adj_matrices[-1] -= sp.eye(n, dtype=np.bool_)

        # Entry x,y is edge x<-y (!)
        adj_matrix = self._bmat_fast(adj_matrices)
        # Entry x,y is edgeid x<-y (!)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape,
        )
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target (i) and source (j) nodes of edges
        data["idnb_i"] = edgeid_to_target
        data["idnb_j"] = edgeid_to_source

        # Indices of triplets k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Indices of triplets that are not i->j->i
        (id3_y_to_d,) = (id3ynb_i != id3ynb_k).nonzero()
        data["id3dnb_i"] = id3ynb_i[id3_y_to_d]
        data["id3dnb_j"] = id3ynb_j[id3_y_to_d]
        data["id3dnb_k"] = id3ynb_k[id3_y_to_d]

        # Edge indices for interactions
        # j->i => k->j
        data["id_expand_kj"] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # j->i => k->j => j->i
        data["id_reduce_ji"] = (
            atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        )
        for key in data.keys():
            data[key] = torch.from_numpy(data[key]).to(self.device)

        return data
