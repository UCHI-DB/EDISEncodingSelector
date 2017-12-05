package edu.uchicago.cs.dist

import org.junit.{Before, Test}
import org.junit.Assert._
import scala.collection.JavaConversions._

/**
  * Created by harper on 5/4/17.
  */
class SlaveTest {

  val reg = new MemoryChannelRegistry()
  var slave: Slave = _

  @Before
  def setup: Unit = {
    reg.channels += ("distribute" -> new MemoryChannel())
    reg.channels += ("collect" -> new MemoryChannel())
    reg.channels += ("heartbeat" -> new MemoryChannel())
    slave = new Slave(reg)
  }

  @Test
  def testStart: Unit = {

    slave.start()

    val task = new TestTask
    val pieces = task.spawn(0)

    pieces.foreach(reg.find("distribute").send)

    // Wait for the task to be executed
    Thread.sleep(100)


    val collect = reg.find("collect").asInstanceOf[MemoryChannel]

    assertEquals(10, collect.content.size())
    collect.content.foreach(item => {
      val piece = item.asInstanceOf[TestTaskPiece]
      assertEquals(piece.id, piece.result)
    })
  }

  @Test
  def testHeartbeat: Unit = {
    slave.start()

    val hb = reg.find("heartbeat").asInstanceOf[MemoryChannel]

    Thread.sleep(1000)
    assertEquals(1, hb.content.size)
    assertEquals(slave.id, hb.content.peek())
  }
}
