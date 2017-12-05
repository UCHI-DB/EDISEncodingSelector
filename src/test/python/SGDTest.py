import unittest
from ndnn import sgd as ns

class MomentumTestCase(unittest.TestCase):
    def test_update(self):
        momentum = ns.Momentum()

class RMSPropTestCase(unittest.TestCase):
    def test_update(self):
        pass

class AdaGradTestCase(unittest.TestCase):
    def test_update(self):
        pass

class AdamTestCase(unittest.TestCase):
    def test_update(self):
        pass

if __name__ == '__main__':
    unittest.main()
