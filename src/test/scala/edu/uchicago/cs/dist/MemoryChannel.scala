package edu.uchicago.cs.dist

import java.util.concurrent.ArrayBlockingQueue

/**
  * Created by harper on 5/4/17.
  */
class MemoryChannel extends Channel {

  val content = new ArrayBlockingQueue[java.io.Serializable](100)

  protected var callback: (java.io.Serializable) => Unit = null

  var id: String = ""

  def send(data: java.io.Serializable): Unit = {
    if (callback == null)
      content.offer(data)
    else {
      callback(data)
    }
  }

  def listen(): java.io.Serializable = content.take()

  def listen(callback: (java.io.Serializable) => Unit) = {
    this.callback = callback
  }

}
