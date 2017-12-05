package edu.uchicago.cs.ndnn

import org.junit.Test
import org.junit.Assert._

class InitPolicyTest {

  //  @Test
  //  def testXavier: Unit = {
  //    var param = Xavier.init(3, 2, 7, 9)
  //    assertArrayEquals(Array(3, 2, 7, 9), param.shape)
  //    assertEquals(0, param.meanNumber().doubleValue(), 0.1)
  //    var n = param.shape.dropRight(1).product
  //    var expectedstd = Math.sqrt(3d / n)
  //    assertEquals(expectedstd, n * param.stdNumber().doubleValue() / (n - 1), 0.1)
  //  }

  @Test
  def testXavier: Unit = {
    val param = Xavier.init(3, 2, 7, 9)
    assertArrayEquals(Array(3, 2, 7, 9), param.shape)
    assertEquals(0, param.meanNumber().doubleValue(), 0.1)
    val n = param.shape.dropRight(1).product
    val range = Math.sqrt(3d / n)
    assertTrue(param.maxNumber().doubleValue() <= range)
    assertTrue(param.minNumber().doubleValue() >= -range)
  }
}