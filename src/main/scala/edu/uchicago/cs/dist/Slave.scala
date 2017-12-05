/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License,
 *
 * Contributors:
 *     Hao Jiang - initial API and implementation
 *
 */

package edu.uchicago.cs.dist

import java.util.UUID
import java.util.concurrent.Executors

import org.slf4j.LoggerFactory

/**
  * Created by harper on 5/3/17.
  */

object Slave extends App {
  new Slave(ChannelRegistry.get).start()
}


class Slave(val registry: ChannelRegistry) {

  val id = UUID.randomUUID().toString

  val receiveChannel = registry.find("distribute")
  val feedbackChannel = registry.find("collect")
  val heartbeatChannel = registry.find("heartbeat")


  val threadPool = Executors.newFixedThreadPool(10)

  val heartbeatThread = new HeartbeatThread(this)

  def start(): Unit = {
    receiveChannel.listen((task: java.io.Serializable) => {
      if (task.isInstanceOf[TaskPiece]) {
        val piece = task.asInstanceOf[TaskPiece]
        threadPool.submit(new TaskRunnable(this, piece))
      }
    })

    heartbeatThread.start()
  }

  def taskDone(piece: TaskPiece): Unit = {
    feedbackChannel.send(piece)
  }
}

class TaskRunnable(val slave: Slave, val piece: TaskPiece) extends Runnable {

  val logger = LoggerFactory.getLogger(getClass())

  override def run(): Unit = {
    try {
      piece.execute()
    } catch {
      case e: Exception => {
        piece.markFail()
      }
    }
    slave.taskDone(piece)
  }
}

class HeartbeatThread(slave: Slave) extends Thread {
  {
    setDaemon(false)
    setName("Heartbeat")
  }

  override def run(): Unit = {
    while (true) {
      slave.heartbeatChannel.send(slave.id)
      // TODO move this to configuration
      Thread.sleep(30000)
    }
  }
}
