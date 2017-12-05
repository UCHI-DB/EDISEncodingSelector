'''
Train a LSTM Model with given input file
'''
import sys
import ndnn.dataset as nds
import ndnn.rnn as ndr
import ndnn.sgd as ndsg
hidden_dim = 200
epoches = 20
dataset = nds.LSTMDataSet(sys.argv[0])

params = []

sgd = ndsg.SGD(0.5, 0.9)
# Train
for epoch in range(epoches):
    for batch in dataset.batches(50):
        graph = ndr.LSTMTrainGraph(dataset.num_char(), hidden_dim)
        if len(params) > 0:
            graph.load(params)
        graph.build(batch)
        graph.update = sgd
        graph.train()
        params = graph.dump()
    sgd.weight_decay()
