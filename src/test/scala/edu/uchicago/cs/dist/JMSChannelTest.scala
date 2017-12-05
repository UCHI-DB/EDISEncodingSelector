package edu.uchicago.cs.dist

import javax.jms.{JMSException, ObjectMessage}

import org.junit.Assert._
import org.junit.Test

import scala.collection.mutable.ArrayBuffer

/**
  * Created by harper on 5/5/17.
  */
class JMSChannelRegistryTest {
  @Test
  def testFind: Unit = {
    try {
      val reg = new JMSChannelRegistry("tcp://localhost:61616")

      val channel = reg.find("a")
    } catch {
      case e: JMSException => {

      }
    }
  }
}

class JMSChannelTest {

  val reg = try {
    new JMSChannelRegistry("tcp://localhost:61616")
  } catch {
    case e: JMSException => {
      null
    }
  }

  @Test
  def testSend: Unit = {
    if (null == reg) {
      return;
    }
    val channel = reg.find("testSend").asInstanceOf[JMSChannel]

    channel.send("test string")
    val message = channel.consumer.receive()

    assertTrue(message.isInstanceOf[ObjectMessage])

    assertEquals("test string", message.asInstanceOf[ObjectMessage].getObject)
  }

  @Test
  def testListen1: Unit = {
    if (null == reg) {
      return;
    }
    val channel = reg.find("testListen1").asInstanceOf[JMSChannel]

    channel.send("test string")

    assertEquals("test string", channel.listen())

    channel.send("test string1")
    channel.send("test string2")
    assertEquals("test string1", channel.listen())
    assertEquals("test string2", channel.listen())
  }

  @Test
  def testListen2: Unit = {
    if (null == reg) {
      return;
    }
    val channel = reg.find("testListen2").asInstanceOf[JMSChannel]

    var buffer = new ArrayBuffer[String]

    channel.listen((a: java.io.Serializable) => {
      buffer += a.toString
    })

    channel.send("test string 1")
    channel.send("test string 2")
    channel.send("test string 3")

    Thread.sleep(1000)

    assertEquals(3, buffer.size)
    assertEquals("test string 1", buffer(0))
    assertEquals("test string 2", buffer(1))
    assertEquals("test string 3", buffer(2))
  }
}
