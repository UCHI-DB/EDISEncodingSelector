package edu.uchicago.cs.ndnn.rnn

import edu.uchicago.cs.ndnn.{Input, NodeEnv, Param}
import org.junit.Assert._
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class LSTMCellTest {

  @Test
  def testBuildCell: Unit = {
    val env = new NodeEnv() {}
    val wf = new Param("wf")
    wf.setEnv(env)
    val bf = new Param("bf")
    bf.setEnv(env)
    val wi = new Param("wi")
    wi.setEnv(env)
    val bi = new Param("bi")
    bi.setEnv(env)
    val wc = new Param("wc")
    wc.setEnv(env)
    val bc = new Param("bc")
    bc.setEnv(env)
    val wo = new Param("wo")
    wo.setEnv(env)
    val bo = new Param("bo")
    bo.setEnv(env)
    val x = new Input("x")
    x.setEnv(env)
    val h = new Input("h")
    h.setEnv(env)
    val c = new Input("c")
    c.setEnv(env)

    val cell = LSTMCell.build(wf, bf, wi, bi, wc, bc, wo, bo, x, h, c)

    wf.set(Nd4j.create(Array(Array(1d, 2, 3, 1, 2, 2d), Array(2d, 0, 1, 2, 1, 0), Array(0d, 1, 1, 0, 1, 2)))
      .transposei().divi(100))
    bf.set(Nd4j.create(Array(3d, 0, 1)).divi(100))
    wi.set(Nd4j.create(Array(Array(2d, 1, 0, 1, 2, 2d), Array(3d, 0, 2, 2, 5, 0), Array(0d, 2, 1, 3, 1, 2)))
      .transposei.divi(100))
    bi.set(Nd4j.create(Array(1d, 2, 0)).divi(100))
    wc.set(Nd4j.create(Array(Array(2d, 2, 3, 2, 2, 2d), Array(2d, 1, 1, 2, 3, 0), Array(1d, 1, 1, 2, 0, 2)))
      .transposei.divi(100))
    bc.set(Nd4j.create(Array(1d, 1, 1)).divi(100))
    wo.set(Nd4j.create(Array(Array(2d, 2, 2, 1, 2, 1d), Array(1d, 1, 1, 1, 1, 0), Array(0d, 0, 1, 0, 1, 0)))
      .transposei.divi(100))
    bo.set(Nd4j.create(Array(1d, 0, 2)).divi(100))

    x.set(Nd4j.create(Array(Array(1d, 0, 1), Array(1d, 1, 0))))
    h.set(Nd4j.zeros(2, 3))
    c.set(Nd4j.ones(2, 3))

    env.forward

    val hout = cell._1.value
    val cout = cell._2.value

    // This value is when concat seq is in,h
    //val cexpect = Nd4j.create(Array(Array(0.54790634, 0.52818812, 0.52007031), Array(0.54047421, 0.5254888, 0.52014528)))
    //val hexpect = Nd4j.create(Array(Array(0.25571027, 0.24441732, 0.24246003), Array(0.25283897, 0.24337213, 0.24129505)))

    // This value is when concat seq is h,in
    val hexpect = Nd4j.create(Array(Array(0.25037327,0.24015875,0.2463134),Array(0.25160626,0.24850409,0.24254814)))
    val cexpect = Nd4j.create(Array(Array(0.54047424,0.52029526,0.533103),Array(0.54047424,0.53881097,0.52029526)))

    assertArrayEquals(Array(2, 3), hout.shape)
    assertArrayEquals(Array(2, 3), cout.shape)

    for (i <- 0 to 1; j <- 0 to 2) {
      assertEquals(hexpect.getDouble(i, j), hout.getDouble(i, j), 0.001)
      assertEquals(cexpect.getDouble(i, j), cout.getDouble(i, j), 0.001)
    }
  }
}