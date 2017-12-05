package edu.uchicago.cs.dist

import scala.collection.mutable

/**
  * Created by harper on 5/4/17.
  */
class MemoryChannelRegistry extends ChannelRegistry {

  val channels = new mutable.HashMap[String, Channel]

  override def find(name: String): Channel = {
    channels.getOrElse(name, null)
  }
}
