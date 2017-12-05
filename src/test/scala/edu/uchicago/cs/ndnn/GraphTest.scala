package edu.uchicago.cs.ndnn

import org.junit.Assert._
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class GraphTest {

  @Test
  def testTrain: Unit = {
    val graph = new Graph(Xavier, new SGD(0.02, 1), new SquareLoss())

    // Setup
    val x = graph.input("x")
    val w = graph.param("W", Array(3, 2))
    val b = graph.param("b", Array(1, 2))(Zero)

    val wx = new DotMul(x, w)
    val add = new Add(wx, b)
    graph.output(add)

    x.set(Nd4j.create(Array(Array(3d, 2d, 1d), Array(5d, 7d, 6d), Array(8d, 2d, 9d), Array(6d, 4d, 1d))))
    graph.expect(Nd4j.create(Array(Array(25d, 31d), Array(79d, 112d), Array(43d, 110d), Array(49d, 54d))))
    var loss = Double.MaxValue
    for (_ <- 0 to 100) {
      loss = graph.train
    }
    /*
     * If this test doesnot converge, most probably because the gradient on node was not cleaned on forward
     */
    assertEquals(0, loss, 0.1)
  }

  @Test
  def testTest: Unit = {

    val graph = new Graph(Xavier, new SGD(0.01), new SoftMaxLogLoss())
    val x = graph.input("x")
    val w = graph.param("W", Array(3, 6))

    val wx = new DotMul(x, w)
    val softmax = new SoftMax(wx)
    graph.output(softmax)

    x.set(Nd4j.create(Array(Array(1d, 2d, 3d), Array(2d, 1d, 9d), Array(5d, 2d, 7d))))
    w.set(Nd4j.create(Array(Array(0.1, 0.5, 2.7, 3.1, 0.2, 0.7), Array(1.1, 2.2, 1.3, 4.2, 5.2, 1.9), Array(3.8, 4.7, 2.2, 0.6, 1.9, 1.3))))

    graph.expect(Nd4j.create(Array(2d, 1d, 4d)).reshape(3, -1))

    assertEquals(22.27 / 3, graph.test._1, 0.01)
  }
}