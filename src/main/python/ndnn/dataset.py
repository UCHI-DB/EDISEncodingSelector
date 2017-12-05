import numpy as np


class Batch:
    def __init__(self, sz, data, label):
        self.size = sz
        self.data = data
        self.expect = label


class DataSet(object):
    def __init__(self, data, label):
        self.datas = data
        self.expects = label
        self.numBatch = 0

    def batches(self, batch_size):
        batch_range = range(0, len(self.datas), batch_size)
        perm = np.random.permutation(len(self.datas)).tolist()
        batches = [perm[idx:idx + batch_size] for idx in batch_range]
        self.numBatch = len(batches)
        for batch_idx in batches:
            data = [self.datas[p] for p in batch_idx]
            expect = [self.expects[p] for p in batch_idx]
            yield Batch(len(batch_idx), np.array(data), np.array(expect))


"""
VarLenDataSet accepts datasets with variable length 
and return data of the same length in one batch
"""


class VarLenDataSet(DataSet):
    def __init__(self, data, label):
        super().__init__(data, label)

        # Group data by length
        group_by_len = {}
        for idx, data in enumerate(self.datas):
            data_len = str(len(data))
            if data_len not in group_by_len:
                group_by_len[data_len] = []
            group_by_len[data_len].append(idx)
        self.group_by_len = list(group_by_len.values())

    def batches(self, batch_size):
        group_perm = np.random.permutation(len(self.group_by_len)).tolist()
        self.numBatch = sum([len(range(0, len(grp), batch_size)) for grp in self.group_by_len])
        for grp_idx in group_perm:
            group = self.group_by_len[grp_idx]
            ing_range = range(0, len(group), batch_size)
            ing_perm = np.random.permutation(len(group)).tolist()
            batches = [ing_perm[idx:idx + batch_size] for idx in ing_range]
            for batch_idx in batches:
                data = [self.datas[group[p]] for p in batch_idx]
                expect = [self.expects[group[p]] for p in batch_idx]
                yield Batch(len(batch_idx), np.array(data), np.array(expect))



