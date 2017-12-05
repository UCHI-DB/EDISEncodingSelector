import numpy as np

from ndnn.dataset import Batch

class LSTMDataSet:
    def __init__(self, vocab_dict, idx_dict, filename):
        self.vocab_dict = vocab_dict
        self.idx_dict = idx_dict
        datas = {}
        lines = open(filename, "rb").readlines()

        for line in lines:
            words = line.decode('utf-8', errors='replace').split()

            idx = np.ndarray((len(words),), dtype=np.int32)
            for i, word in enumerate(words):
                if word not in self.vocab_dict:
                    raise Exception()
                idx[i] = self.vocab_dict[word]
            keylen = len(idx)
            if keylen not in datas:
                datas[keylen] = []
            datas[keylen].append(idx)

        self.datas = list(datas.values())

    def translate_to_str(self, numarray):
        return ' '.join([self.idx_dict[n] for n in numarray])

    def num_batch(self):
        return self.numbatch

    def batches(self, batch_size):
        self.numbatch = np.sum([np.ceil(len(item) / batch_size) for item in self.datas])

        perm = np.random.permutation(len(self.datas))

        for p in perm:
            batch_base = self.datas[p]
            subperm = np.random.permutation(len(batch_base))
            batch_range = range(0, len(batch_base), batch_size)

            batch_idices = [subperm[idx:idx + batch_size] for idx in batch_range]

            for batch_idx in batch_idices:
                item = [batch_base[idx] for idx in batch_idx]
                yield Batch(len(item), np.array(item), None)

class S2SDataSet:
    def __init__(self, vocab_dict, idx_dict, filename):
        self.vocab_dict = vocab_dict
        self.idx_dict = idx_dict
        datas = {}
        lines = open(filename, "rb").readlines()

        for line in lines:
            sentences = line.decode('utf-8', errors='replace').split('\t')

            psent = sentences[0]
            csent = sentences[1]

            pwordIdx = [vocab_dict[word] for word in psent.split()]
            cwordIdx = [vocab_dict[word] for word in csent.split()]

            plen = len(pwordIdx)
            clen = len(cwordIdx)

            key = (plen, clen)
            if key not in datas:
                datas[key] = []
            datas[key].append([pwordIdx, cwordIdx])

        self.datas = list(datas.values())
        self.numbatch = 0

    def translate_to_str(self, numarray):
        return ' '.join([self.idx_dict[n] for n in numarray])

    def num_batch(self):
        return self.numbatch

    def batches(self, batch_size):
        self.numbatch = np.sum([np.ceil(len(item) / batch_size) for item in self.datas])

        perm = np.random.permutation(len(self.datas))

        for p in perm:
            batch_base = self.datas[p]
            subperm = np.random.permutation(len(batch_base))
            batch_range = range(0, len(batch_base), batch_size)

            batch_idices = [subperm[idx:idx + batch_size] for idx in batch_range]

            for batch_idx in batch_idices:
                item1 = []
                item2 = []
                for idx in batch_idx:
                    item1.append(batch_base[idx][0])
                    item2.append(batch_base[idx][1])
                yield Batch(len(batch_idx), [np.array(item1), np.array(item2)], None)
