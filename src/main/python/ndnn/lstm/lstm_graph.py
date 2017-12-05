import numpy as np

from ndnn.lstm.lstm_loss import LogLoss, HingeLoss, HingeLossOutput
from ndnn.lstm.lstm_node import Attention
from ndnn.graph import Graph
from ndnn.init import Xavier, Zero
from ndnn.node import Concat, Sigmoid, Add, Dot, Tanh, Mul, Collect, Embed, SoftMax, MDEmbed, Average, ArgMax


class LSTMGraph(Graph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update)

        self.dict_size = dict_size
        self.hidden_dim = hidden_dim

        self.h0 = self.input()
        self.c0 = self.input()

        self.embed = self.param_of([dict_size, hidden_dim], Xavier())
        self.wf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bf = self.param_of([hidden_dim], Zero())
        self.wi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bi = self.param_of([hidden_dim], Zero())
        self.wc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bc = self.param_of([hidden_dim], Zero())
        self.wo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.bo = self.param_of([hidden_dim], Zero())
        self.v2c = self.param_of([hidden_dim, dict_size], Xavier())

        self.resetNum = len(self.nodes)

    def lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.wf), self.bf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.wi), self.bi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.wc), self.bc))
        o_temp = Sigmoid(Add(Dot(concat, self.wo), self.bo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next


class LogGraph(LSTMGraph):
    def __init__(self, update, dict_size, hidden_dim):
        super().__init__(LogLoss(), update, dict_size, hidden_dim)

    def build_graph(self, batch):
        self.reset()
        # Build Computation Graph according to length
        bsize, length = batch.data.shape

        self.h0.value = np.zeros([bsize, self.hidden_dim])
        self.c0.value = np.zeros([bsize, self.hidden_dim])

        h = self.h0
        c = self.c0
        outputs = []
        for idx in range(length - 1):
            in_i = self.input()
            in_i.value = batch.data[:, idx]  # Get value from batch
            x = Embed(in_i, self.embed)
            h, c = self.lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.v2c))
            outputs.append(out_i)
        self.output(Collect(outputs))
        self.expect(batch.data[:, 1:])


class HingeGraph(LSTMGraph):
    def __init__(self, update, dict_size, hidden_dim, num_neg_sample, sep_embed):
        super().__init__(HingeLoss(), update, dict_size, hidden_dim)
        self.num_neg_sample = num_neg_sample

        self.neg_sample = self.input()
        if sep_embed:
            self.sample_embed = self.param_of([dict_size, hidden_dim], Xavier())
        else:
            self.sample_embed = self.embed

        self.resetNum = len(self.nodes)

    def build_graph(self, batch):
        self.reset()
        # Build Computation Graph according to length
        bsize, length = batch.data.shape

        if self.num_neg_sample == -1:
            negSampleIdx = range(self.dict_size)
        else:
            negSampleIdx = np.array([np.random.randint(low=0, high=self.dict_size) for i in range(self.num_neg_sample)])
        self.neg_sample.value = np.int32(negSampleIdx)

        self.h0.value = np.zeros([bsize, self.hidden_dim])
        self.c0.value = np.zeros([bsize, self.hidden_dim])

        h = self.h0
        c = self.c0
        outputs = []
        for idx in range(length - 1):
            in_i = self.input()
            in_i.value = batch.data[:, idx]  # Get value from batch
            x = Embed(in_i, self.embed)
            h, c = self.lstm_cell(x, h, c)

            outputs.append(h)

        self.output(HingeLossOutput(Collect(outputs), self.sample_embed, self.neg_sample))
        self.expect(batch.data[:, 1:])


class LSTMEncodeGraph(Graph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update)

        self.dict_size = dict_size
        self.hidden_dim = hidden_dim

        self.h0 = self.input()
        self.c0 = self.input()

        self.eembed = self.param_of([dict_size, hidden_dim], Xavier())
        self.ewf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.ebf = self.param_of([hidden_dim], Zero())
        self.ewi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.ebi = self.param_of([hidden_dim], Zero())
        self.ewc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.ebc = self.param_of([hidden_dim], Zero())
        self.ewo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.ebo = self.param_of([hidden_dim], Zero())
        self.ev2c = self.param_of([hidden_dim, dict_size], Xavier())

        self.dembed = self.param_of([dict_size, hidden_dim], Xavier())
        self.dwf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbf = self.param_of([hidden_dim], Zero())
        self.dwi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbi = self.param_of([hidden_dim], Zero())
        self.dwc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbc = self.param_of([hidden_dim], Zero())
        self.dwo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbo = self.param_of([hidden_dim], Zero())
        self.dv2c = self.param_of([hidden_dim, dict_size], Xavier())

        self.resetNum = len(self.nodes)

    def enc_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.ewf), self.ebf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.ewi), self.ebi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.ewc), self.ebc))
        o_temp = Sigmoid(Add(Dot(concat, self.ewo), self.ebo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def dec_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.dwf), self.dbf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.dwi), self.dbi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.dwc), self.dbc))
        o_temp = Sigmoid(Add(Dot(concat, self.dwo), self.dbo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def build_graph(self, batch):
        enc_data = batch.data[0]
        dec_data = batch.data[1]
        self.reset()

        bsize, enc_length = enc_data.shape
        dec_length = dec_data.shape[1]

        outputs = []

        # Build Encode Graph
        self.h0.value = np.zeros([bsize, self.hidden_dim])
        self.c0.value = np.zeros([bsize, self.hidden_dim])

        h = self.h0
        c = self.c0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.eembed)
            h, c = self.enc_lstm_cell(x, h, c)
            # out_i = SoftMax(Dot(h, graph.ev2c))
            # outputs.append(out_i)

        self.encoded_h = h
        self.encoded_c = c

        # Build Decode Graph
        for idx in range(dec_length - 1):
            in_i = self.input()
            in_i.value = dec_data[:, idx]
            x = Embed(in_i, self.dembed)
            h, c = self.dec_lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.dv2c))
            outputs.append(out_i)

        self.output(Collect(outputs))
        self.expect(dec_data[:, 1:])

    def encode_result(self):
        # return np.concatenate((self.encoded_h.value, self.encoded_c.value), axis=1)
        return self.encoded_c.value


class LSTMDecodeGraph(LSTMEncodeGraph):
    def __init__(self, loss, dict_size, hidden_dim, predict_len):
        super().__init__(loss, None, dict_size, hidden_dim)
        self.predict_len = predict_len

    def build_graph(self, batch):
        enc_data = batch.data
        self.reset()

        bsize, enc_length = enc_data.shape

        outputs = []

        # Build Encode Graph
        self.h0.value = np.zeros([bsize, self.hidden_dim])
        self.c0.value = np.zeros([bsize, self.hidden_dim])

        h = self.h0
        c = self.c0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.eembed)
            h, c = self.enc_lstm_cell(x, h, c)
            # out_i = SoftMax(Dot(h, graph.ev2c))
            # outputs.append(out_i)

        self.encoded_h = h
        self.encoded_c = c

        # Build Decode Graph

        decode_in = self.input()
        decode_in.value = np.zeros([bsize])
        decode_embed = Embed(decode_in, self.dembed)
        x = decode_embed
        for idx in range(self.predict_len):
            h, c = self.dec_lstm_cell(x, h, c)
            out_i = ArgMax(SoftMax(Dot(h, self.dv2c)))
            outputs.append(out_i)
            x = Embed(out_i, self.dembed)
        self.output(Collect(outputs))
        self.expect(np.zeros([bsize, self.predict_len]))


class BiLSTMEncodeGraph(Graph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update)

        half_dim = int(hidden_dim / 2)

        self.dict_size = dict_size
        self.hidden_dim = hidden_dim
        self.half_dim = half_dim

        self.feh0 = self.input()
        self.fec0 = self.input()

        self.feembed = self.param_of([dict_size, half_dim], Xavier())
        self.fewf = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febf = self.param_of([half_dim], Zero())
        self.fewi = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febi = self.param_of([half_dim], Zero())
        self.fewc = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febc = self.param_of([half_dim], Zero())
        self.fewo = self.param_of([2 * half_dim, half_dim], Xavier())
        self.febo = self.param_of([half_dim], Zero())
        self.fev2c = self.param_of([half_dim, dict_size], Xavier())

        self.beh0 = self.input()
        self.bec0 = self.input()

        self.beembed = self.param_of([dict_size, half_dim], Xavier())
        self.bewf = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebf = self.param_of([half_dim], Zero())
        self.bewi = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebi = self.param_of([half_dim], Zero())
        self.bewc = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebc = self.param_of([half_dim], Zero())
        self.bewo = self.param_of([2 * half_dim, half_dim], Xavier())
        self.bebo = self.param_of([half_dim], Zero())
        self.bev2c = self.param_of([half_dim, dict_size], Xavier())

        self.dembed = self.param_of([dict_size, hidden_dim], Xavier())
        self.dwf = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbf = self.param_of([hidden_dim], Zero())
        self.dwi = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbi = self.param_of([hidden_dim], Zero())
        self.dwc = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbc = self.param_of([hidden_dim], Zero())
        self.dwo = self.param_of([2 * hidden_dim, hidden_dim], Xavier())
        self.dbo = self.param_of([hidden_dim], Zero())
        self.dv2c = self.param_of([hidden_dim, dict_size], Xavier())

        self.resetNum = len(self.nodes)

    def fenc_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.fewf), self.febf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.fewi), self.febi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.fewc), self.febc))
        o_temp = Sigmoid(Add(Dot(concat, self.fewo), self.febo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def benc_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.bewf), self.bebf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.bewi), self.bebi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.bewc), self.bebc))
        o_temp = Sigmoid(Add(Dot(concat, self.bewo), self.bebo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def dec_lstm_cell(self, x, h, c):
        concat = Concat(h, x)

        # Forget Gate
        f_gate = Sigmoid(Add(Dot(concat, self.dwf), self.dbf))
        # Input Gate
        i_gate = Sigmoid(Add(Dot(concat, self.dwi), self.dbi))
        # Temp Vars
        c_temp = Tanh(Add(Dot(concat, self.dwc), self.dbc))
        o_temp = Sigmoid(Add(Dot(concat, self.dwo), self.dbo))

        # Output
        c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
        h_next = Mul(o_temp, Tanh(c_next))
        return h_next, c_next

    def build_graph(self, batch):
        enc_data = batch.data[0]
        dec_data = batch.data[1]
        self.reset()

        bsize, enc_length = enc_data.shape
        dec_length = dec_data.shape[1]

        outputs = []

        # Build Fwd Encode Graph
        self.feh0.value = np.zeros([bsize, self.half_dim])
        self.fec0.value = np.zeros([bsize, self.half_dim])

        fh = self.feh0
        fc = self.fec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.feembed)
            fh, fc = self.fenc_lstm_cell(x, fh, fc)

        # Build Bwd Encode Graph
        self.beh0.value = np.zeros([bsize, self.half_dim])
        self.bec0.value = np.zeros([bsize, self.half_dim])

        bh = self.beh0
        bc = self.bec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, enc_length - 1 - idx]  # Get value from batch
            x = Embed(in_i, self.beembed)
            bh, bc = self.benc_lstm_cell(x, bh, bc)

        # Build Decode Graph
        h = Concat(fh, bh)
        c = Concat(fc, bc)

        self.encoded_h = h
        self.encoded_c = c

        for idx in range(dec_length - 1):
            in_i = self.input()
            in_i.value = dec_data[:, idx]
            x = Embed(in_i, self.dembed)
            h, c = self.dec_lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.dv2c))
            outputs.append(out_i)

        self.output(Collect(outputs))
        self.expect(dec_data[:, 1:])

    def encode_result(self):
        # return np.concatenate((self.encoded_h.value, self.encoded_c.value), axis=1)
        return self.encoded_c.value


class BiLSTMDecodeGraph(BiLSTMEncodeGraph):
    def __init__(self, loss, dict_size, hidden_dim, predict_len):
        super().__init__(loss, None, dict_size, hidden_dim)
        self.predict_len = predict_len

    def build_graph(self, batch):
        enc_data = batch.data
        self.reset()

        bsize, enc_length = enc_data.shape

        outputs = []

        # Build Fwd Encode Graph
        self.feh0.value = np.zeros([bsize, self.half_dim])
        self.fec0.value = np.zeros([bsize, self.half_dim])

        fh = self.feh0
        fc = self.fec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.feembed)
            fh, fc = self.fenc_lstm_cell(x, fh, fc)

        # Build Bwd Encode Graph
        self.beh0.value = np.zeros([bsize, self.half_dim])
        self.bec0.value = np.zeros([bsize, self.half_dim])

        bh = self.beh0
        bc = self.bec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, enc_length - 1 - idx]  # Get value from batch
            x = Embed(in_i, self.beembed)
            bh, bc = self.benc_lstm_cell(x, bh, bc)

        # Build Decode Graph
        h = Concat(fh, bh)
        c = Concat(fc, bc)

        self.encoded_h = h
        self.encoded_c = c

        # Build Decode Graph

        decode_in = self.input()
        decode_in.value = np.zeros([bsize])
        decode_embed = Embed(decode_in, self.dembed)
        x = decode_embed
        for idx in range(self.predict_len):
            h, c = self.dec_lstm_cell(x, h, c)
            out_i = ArgMax(SoftMax(Dot(h, self.dv2c)))
            outputs.append(out_i)
            x = Embed(out_i, self.dembed)
        self.output(Collect(outputs))
        self.expect(np.zeros([bsize, self.predict_len]))


class AttentionGraph(BiLSTMEncodeGraph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update, dict_size, hidden_dim)

    def build_graph(self, batch):
        enc_data = batch.data[0]
        dec_data = batch.data[1]
        self.reset()

        bsize, enc_length = enc_data.shape
        dec_length = dec_data.shape[1]

        outputs = []

        fwd_encode_result = [None] * enc_length
        bwd_encode_result = [None] * enc_length

        # Build Fwd Encode Graph
        self.feh0.value = np.zeros([bsize, self.half_dim])
        self.fec0.value = np.zeros([bsize, self.half_dim])

        fh = self.feh0
        fc = self.fec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.feembed)
            fh, fc = self.fenc_lstm_cell(x, fh, fc)
            fwd_encode_result[idx] = fh

        # Build Bwd Encode Graph
        self.beh0.value = np.zeros([bsize, self.half_dim])
        self.bec0.value = np.zeros([bsize, self.half_dim])

        bh = self.beh0
        bc = self.bec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, enc_length - 1 - idx]  # Get value from batch
            x = Embed(in_i, self.beembed)
            bh, bc = self.benc_lstm_cell(x, bh, bc)
            bwd_encode_result[enc_length - 1 - idx] = bh

        # Build Decode Graph
        h = Concat(fh, bh)
        # c = Concat(fc, bc)

        self.encoded_h = h
        # self.encoded_c = c

        encode_result = []
        for idx in range(enc_length):
            encode_result.append(Concat(fwd_encode_result[idx], bwd_encode_result[idx]))
        encode_state = Collect(encode_result)

        for idx in range(dec_length - 1):
            in_i = self.input()
            in_i.value = dec_data[:, idx]
            x = Embed(in_i, self.dembed)

            c = Attention(encode_state, h)

            h, c = self.dec_lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.dv2c))
            outputs.append(out_i)

        self.output(Collect(outputs))
        self.expect(dec_data[:, 1:])


class AttentionDecodeGraph(BiLSTMEncodeGraph):
    def __init__(self, loss, dict_size, hidden_dim, predict_len):
        super().__init__(loss, None, dict_size, hidden_dim)
        self.predict_len = predict_len

    def build_graph(self, batch):
        enc_data = batch.data
        self.reset()

        bsize, enc_length = enc_data.shape

        outputs = []

        fwd_encode_result = [None] * enc_length
        bwd_encode_result = [None] * enc_length

        # Build Fwd Encode Graph
        self.feh0.value = np.zeros([bsize, self.half_dim])
        self.fec0.value = np.zeros([bsize, self.half_dim])

        fh = self.feh0
        fc = self.fec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, idx]  # Get value from batch
            x = Embed(in_i, self.feembed)
            fh, fc = self.fenc_lstm_cell(x, fh, fc)
            fwd_encode_result[idx] = fh

        # Build Bwd Encode Graph
        self.beh0.value = np.zeros([bsize, self.half_dim])
        self.bec0.value = np.zeros([bsize, self.half_dim])

        bh = self.beh0
        bc = self.bec0
        for idx in range(enc_length):
            in_i = self.input()
            in_i.value = enc_data[:, enc_length - 1 - idx]  # Get value from batch
            x = Embed(in_i, self.beembed)
            bh, bc = self.benc_lstm_cell(x, bh, bc)
            bwd_encode_result[enc_length - 1 - idx] = bh

        # Build Decode Graph
        h = Concat(fh, bh)
        # c = Concat(fc, bc)

        self.encoded_h = h
        # self.encoded_c = c

        encode_result = []
        for idx in range(enc_length):
            encode_result.append(Concat(fwd_encode_result[idx], bwd_encode_result[idx]))
        encode_state = Collect(encode_result)

        init = self.input()
        init.value = np.zeros([bsize])
        x = Embed(init, self.dembed)
        for idx in range(self.predict_len):
            c = Attention(encode_state, h)

            h, c = self.dec_lstm_cell(x, h, c)
            out_i = ArgMax(SoftMax(Dot(h, self.dv2c)))
            outputs.append(out_i)
            x = Embed(out_i, self.dembed)
        self.output(Collect(outputs))
        self.expect(np.zeros([bsize, self.predict_len]))


class BowEncodeGraph(LSTMGraph):
    def __init__(self, loss, update, dict_size, hidden_dim):
        super().__init__(loss, update, dict_size, hidden_dim)

    def bow_encode(self, data):
        self.h0.value = None
        self.c0.value = None
        h0c0 = self.input()
        h0c0.value = data

        emb = MDEmbed(h0c0, self.embed)
        avg = Average(emb)
        self.encoded_h = avg
        self.encoded_c = avg
        return avg, avg

    def build_graph(self, batch):
        data = batch.data[1]
        self.reset()
        # Build Computation Graph according to length
        bsize, length = data.shape

        h, c = self.bow_encode(batch.data[0])

        outputs = []
        for idx in range(length - 1):
            in_i = self.input()
            in_i.value = data[:, idx]  # Get value from batch
            x = Embed(in_i, self.embed)
            h, c = self.lstm_cell(x, h, c)
            out_i = SoftMax(Dot(h, self.v2c))
            outputs.append(out_i)
        self.output(Collect(outputs))
        self.expect(data[:, 1:])

    def encode_result(self):
        # return np.concatenate((self.encoded_h.value, self.encoded_c.value), axis=1)
        return self.encoded_c.value
