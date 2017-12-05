package edu.uchicago.cs.ndnn


import scala.collection.mutable.{ArrayBuffer, HashSet}
import org.junit.Assert.assertEquals
import org.junit.Test

class DummyDataset extends DefaultDataset {
  def permuteIdx4Test = permuteIdx

  def load(): Array[Data] = {

    val data = new ArrayBuffer[Data]
    for (i <- 0 until 100) {
      data += new DefaultData((0 to 11).map(_ => i.toDouble).toArray, i.toDouble)
    }
    data.toArray
  }
}

class DatasetTest {

  @Test
  def testBatches: Unit = {
    val ds = new DummyDataset

    var array = ds.batches(27).toArray
    assertEquals(4, array.length)

    assertEquals(27, array(0).size)
    assertEquals(27, array(1).size)
    assertEquals(27, array(2).size)
    assertEquals(19, array(3).size)

    val distinct = new HashSet[Double]()
    array.foreach { batch => {
      (0 until batch.size).foreach {
        i => {
          val fromdata = batch.feature(0).getDouble(i, 0)
          distinct += fromdata
          for (k <- 0 to 11) {
            assertEquals(batch.groundTruth.getDouble(i), batch.feature(0).getDouble(i, k), 0.01)
          }
        }
      }
    }
    }
    assertEquals(100, distinct.size)
    assertEquals(0, distinct.min, 0.01)
    assertEquals(99, distinct.max, 0.01)

    distinct.clear()

    array = ds.batches(31).toArray
    assertEquals(4, array.length)

    assertEquals(31, array(0).size)
    assertEquals(31, array(1).size)
    assertEquals(31, array(2).size)
    assertEquals(7, array(3).size)

    array.foreach { batch => {
      (0 until batch.size).foreach {
        i => {
          val fromdata = batch.feature(0).getDouble(i, 0)
          distinct += fromdata
          assertEquals(fromdata, batch.groundTruth.getDouble(i), 0.01)
          for (j <- 0 to 11) {
            assertEquals(fromdata, batch.feature(0).getDouble(i, j), 0.01)
          }

        }
      }
    }
    }
    assertEquals(100, distinct.size)
    assertEquals(0, distinct.min, 0.01)
    assertEquals(99, distinct.max, 0.01)
  }

  @Test
  def testSplit: Unit = {

    val data = (0 until 10000).map(i => Array.fill(100)(i.toDouble)).toArray
    val label = (0 until 10000).map(_.toDouble).toArray

    val dataset = new SplitDatasetForTest(data, label)

    val splitds = dataset.split(Seq(0.7, 0.15, 0.1))

    assertEquals(7000, splitds(0).dataSize)
    assertEquals(1500, splitds(1).dataSize)
    assertEquals(1500, splitds(2).dataSize)
  }
}

class SplitDatasetForTest(value: Array[Array[Double]], label: Array[Double])
  extends DefaultDataset {
  override def load(): Array[Data] = {
    value.zip(label).map(pair => new DefaultData(pair._1, pair._2))
  }
}

