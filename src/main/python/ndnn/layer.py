import ndnn.node as ndn
import ndnn.init as ndi


class Layer(object):
    def __init__(self):
        self.params = []
        self.inputs = []
        self.outputs = []

    def input(self):
        x = ndn.Input()
        self.inputs.append(x)
        return x

    def param(self):
        param = ndn.Param()
        self.params.append(param)
        return param

    def param_of(self, shape, init=ndi.Xavier()):
        param = ndn.Param()
        param.value = init.apply(shape)
        self.params.append(param)
        return param

    def output(self, node):
        self.outputs.append(node)

    def forward(self):
        for x in self.inputs:
            x.forward(x)
        for p in self.params:
            p.forward(p)

    def backward(self):
        for o in self.outputs:
            o.backward(o)


class RecursiveLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        ite = self.num_iteration()

        for i in range(ite):
            self.ite_forward(i)
            for x in self.inputs:
                x.forward(x)
            self.ite_done(i)

    def backward(self):
        ite = self.num_iteration()

        for i in range(ite):
            self.ite_backward(i)
            for x in self.outputs:
                x.backward(x)
            self.ite_done(i)
