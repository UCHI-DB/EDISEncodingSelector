package edu.uchicago.cs.ndnn

import org.junit.Assert._
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j


class IndexerTest {

  @Test
  def testGet: Unit = {
    val a = Nd4j.create(Array(Array(3d, 4d, 5d), Array(1d, 6d, 7d)))
    val b = Nd4j.create(Array(2d, 1d)).reshape(2, 1)

    val fetched = Index.get(a, b)
    assertArrayEquals(Array(2, 1), fetched.shape)
    assertEquals(5d, fetched.getDouble(0, 0), 0.001)
    assertEquals(6d, fetched.getDouble(1, 0), 0.001)
  }

  @Test
  def testPut: Unit = {
    val a = Nd4j.createUninitialized(Array(2, 3)).assign(0)
    val b = Nd4j.create(Array(2d, 1d)).reshape(2, 1)
    val value = Nd4j.create(Array(3d, 5d)).reshape(2, 1)

    Index.put(a, b, value)

    assertEquals(3d, a.getDouble(0, 2), 0.001)
    assertEquals(5d, a.getDouble(1, 1), 0.001)
  }
}