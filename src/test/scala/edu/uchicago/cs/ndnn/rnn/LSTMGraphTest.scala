package edu.uchicago.cs.ndnn.rnn

import org.junit.Test

class LSTMGraphTest {
  @Test
  def testGraphRun: Unit = {
    val file = "src/test/resource/rnn/lstm_sample_ds"
    val batchsize = 50
    val ds = new LSTMDataset(file, new WordTokenizer(), "<s>", "</s>")
    val batch = ds.batches(batchsize).next()
    val data = batch.feature(0)
    val hiddendim = 200

    val graph = new LSTMGraph(ds.dictSize, hiddendim)
    graph.build(batch)
    graph.train
  }

}