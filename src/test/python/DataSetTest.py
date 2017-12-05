import unittest
import numpy as np

import ndnn.dataset as nd


class DataSetTest(unittest.TestCase):
    def test_batch(self):
        data = [[1, 3, 5, 6, 7], [2, 2, 8, 8, 1], [3, 3, 6, 0, 9], [4, 0, 4, 2, 1], [5, 6, 1, 0, 2]]
        label = [1, 2, 3, 4, 5]
        ds = nd.DataSet(data, label)

        item_counter = 0
        counter = 0
        for batch in ds.batches(2):
            item_counter += batch.size
            data = batch.data
            label = batch.expect
            if counter != 2:
                self.assertEqual(2, data.shape[0])
            else:
                self.assertEqual(1, data.shape[0])
            self.assertTrue(np.array_equal(data[:, 0], label))
            counter += 1
        self.assertEqual(3, ds.numBatch)
        self.assertEqual(5, item_counter)


class VarLenDataSetTest(unittest.TestCase):
    def test_batch(self):
        data = [[1, 2, 3, 4], [2, 1, 6], [3, 5, 8], [4, 2, 1, 8, 9, 10], [5, 2, 6, 0, 0, 1], [6, 3, 2, 7],
                [7, 0, 9, 2, 1, 8, 4], [8, 1, 9], [9, 2, 3, 4, 1, 0], [10, 2, 1, 4, 2, 1]]
        label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ds = nd.VarLenDataSet(data, label)

        item_counter = 0
        counter = 0
        for batch in ds.batches(2):
            item_counter += batch.size
            counter += 1
            self.assertTrue(type(batch.data) is np.ndarray)
            self.assertTrue(type(batch.expect) is np.ndarray)
            self.assertEqual(batch.size, batch.data.shape[0])
            self.assertEqual(batch.size, batch.expect.shape[0])
            self.assertTrue(np.array_equal(batch.data[:, 0], batch.expect))

        self.assertEqual(10, item_counter)
        self.assertEqual(6, counter)
        self.assertEqual(6, ds.numBatch)


if __name__ == '__main__':
    unittest.main()
