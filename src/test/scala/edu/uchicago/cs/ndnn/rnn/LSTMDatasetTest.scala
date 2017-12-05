package edu.uchicago.cs.ndnn.rnn

import org.junit.Assert._
import org.junit.Test

class LSTMDatasetTest {

  @Test
  def testLoad: Unit = {
    val file = "src/test/resource/rnn/lstm_sample_ds"
    val ds = new LSTMDataset(file, new WordTokenizer(), "<s>", "</s>")

    assertEquals(8, ds.dataSize)
    assertEquals(22, ds.dictSize)

    val batches = ds.batches(3).toArray
    assertEquals(5, batches.length)

    assertEquals(8, batches.map(_.size).sum)
  }
}

class S2SDatasetTest {
  @Test
  def testLoad: Unit = {
    val file = "src/test/resource/rnn/s2s_sample_ds"
    val ds = new S2SDataset(file, new WordTokenizer(), "<s>", "</s>")

    assertEquals(6, ds.dataSize)
    assertEquals(10, ds.encDictSize)
    assertEquals(13, ds.decDictSize)

    val batches = ds.batches(5).toArray

    assertEquals(6, batches.length)

    batches.foreach(batch => {
      val enc = batch.feature(0)
      val dec = batch.feature(1)
      val eshape = enc.shape()
      val dshape = dec.shape()
      assertEquals(1, eshape(0))
      assertEquals(1, dshape(0))
    })
  }
}