package edu.uchicago.cs.dist

import org.junit.{Before, Test}
import org.junit.Assert._

/**
  * Created by harper on 5/4/17.
  */
class MasterTest {

  var master: Master = _

  var reg = new MemoryChannelRegistry()

  @Before
  def setup: Unit = {
    reg.channels += ("distribute" -> new MemoryChannel())
    reg.channels += ("collect" -> new MemoryChannel())
    reg.channels += ("heartbeat" -> new MemoryChannel())
    master = new Master(reg)
  }

  @Test
  def testSubmit: Unit = {

    val task = new TestTask()

    master.submit(task)

    val distchannel = reg.find("distribute").asInstanceOf[MemoryChannel]

    assertEquals(10, distchannel.content.size())
    assertEquals(1, master.taskLog.size)

    assertEquals(10, master.taskLog.get(task.id).get._2)
  }

  @Test
  def testCollect: Unit = {

    val task = new TestTask()
    master.submit(task)
    val distchannel = reg.find("distribute")
    val colchannel = reg.find("collect")

    for (i <- 0 to 9) {
      val piece = distchannel.listen().asInstanceOf[TestTaskPiece]
      piece.result = piece.id
      assertTrue(master.taskLog.contains(task.id))
      colchannel.send(piece)
    }
    assertFalse(master.taskLog.contains(task.id))

    assertEquals(10, task.done.size)
  }

  @Test
  def testHeartbeat: Unit = {
    val hbchannel = reg.find("heartbeat")
    for (i <- 0 to 9) {
      hbchannel.send(i.toString)
    }

    assertEquals(10, master.slaves.size)
  }
}
