package edu.uchicago.cs.ndnn

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class BroadcastTest {

  @Test
  def testDiff: Unit = {
    val a = Nd4j.createUninitialized(Array(4, 2, 3, 5, 9))
    val b = Nd4j.createUninitialized(Array(4, 5, 8))

    val diff = Broadcast.diff(a.shape, b.shape)

    assertArrayEquals(Array(2), diff._1)
    assertArrayEquals(Array(0, 1, 4), diff._2)
  }

  @Test
  def testAdd: Unit = {
    var a = Nd4j.create(Array(1d, 2, 3, 4, 5, 6, 7, 8)).reshape(2, 4)
    var b = Nd4j.create(Array(1d, 2, 3, 4, 5, 6, 7, 8)).reshape(2, 4)

    var expected = Nd4j.create(Array(2d, 4, 6, 8, 10, 12, 14, 16)).reshape(2, 4)

    var actual = Broadcast.add(a, b)

    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    b = Nd4j.create(Array(1d, 2, 3, 4))
    expected = Nd4j.create(Array(2d, 4, 6, 8, 6, 8, 10, 12)).reshape(2, 4)
    actual = Broadcast.add(a, b)
    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    b = b.transpose()
    actual = Broadcast.add(a, b)
    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    a = a.reshape(2, 2, 2)
    b = Nd4j.create(Array(1d, 2))
    expected = Nd4j.create(Array(2d, 4, 4, 6, 6, 8, 8, 10)).reshape(2, 2, 2)
    actual = Broadcast.add(a, b)
    for (i <- 0 to 1; j <- 0 to 1; k <- 0 to 1)
      assertEquals(actual.getDouble(i, j, k), expected.getDouble(i, j, k), 0.0001)

  }

  @Test
  def testMul: Unit = {
    var a = Nd4j.create(Array(1d, 2, 3, 4, 5, 6, 7, 8)).reshape(2, 4)
    var b = Nd4j.create(Array(1d, 2, 3, 4, 5, 6, 7, 8)).reshape(2, 4)

    var expected = Nd4j.create(Array(1d, 4, 9, 16, 25, 36, 49, 64)).reshape(2, 4)

    var actual = Broadcast.mul(a, b)

    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    b = Nd4j.create(Array(1d, 2, 3, 4))
    expected = Nd4j.create(Array(1d, 4, 9, 16, 5, 12, 21, 32)).reshape(2, 4)
    actual = Broadcast.mul(a, b)
    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    b = b.transpose()
    actual = Broadcast.mul(a, b)
    for (i <- 0 to 1; j <- 0 to 3)
      assertEquals(actual.getDouble(i, j), expected.getDouble(i, j), 0.0001)

    a = a.reshape(2, 2, 2)
    b = Nd4j.create(Array(1d, 2))
    expected = Nd4j.create(Array(1d, 4, 3, 8, 5, 12, 7, 16)).reshape(2, 2, 2)
    actual = Broadcast.mul(a, b)
    for (i <- 0 to 1; j <- 0 to 1; k <- 0 to 1)
      assertEquals(actual.getDouble(i, j, k), expected.getDouble(i, j, k), 0.0001)

  }
}