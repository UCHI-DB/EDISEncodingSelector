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

import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * Created by harper on 5/3/17.
  */

object Master {
  val instance = new Master(ChannelRegistry.get)
}

class Master(val registry: ChannelRegistry) {

  val logger = LoggerFactory.getLogger(getClass())

  val heartbeatChannel = registry.find("heartbeat")
  val distributeChannel = registry.find("distribute")
  val collectChannel = registry.find("collect")

  val slaves = new mutable.HashMap[String, Long]

  val taskLog = new mutable.HashMap[String, (Task, Int)]

  {
    collectChannel.listen((data: java.io.Serializable) => {
      if (data.isInstanceOf[TaskPiece]) {
        collect(data.asInstanceOf[TaskPiece])
      }
    })

    heartbeatChannel.listen((data: java.io.Serializable) => {
      slaves.synchronized {
        slaves += ((data.toString, System.currentTimeMillis()))
      }
    })

    new Thread() {
      {
        setName("SlaveCleaner")
      }

      override def run(): Unit = {
        // TODO Use configuration to replace these constants
        slaves.synchronized({
          val current = System.currentTimeMillis()
          slaves.foreach(item => {
            if (current - item._2 > 60000) {
              slaves.remove(item._1)
            }
          })
        })
        Thread.sleep(5000)
      }
    }.start()
  }

  def submit(task: Task) = {
    val numSlaves = slaves.size
    val pieces = task.spawn(numSlaves)
    taskLog += ((task.id -> (task, pieces.length)))
    pieces.foreach(distributeChannel.send)
    logger.debug("TaskPiece distributed: %s".format(task.id))
  }

  def collect(piece: TaskPiece) = {
    logger.debug("TaskPiece collected: %s:%s".format(piece.parentId, piece.id))

    this.synchronized({
      taskLog.get(piece.parentId) match {
        case None => {
          logger.error("Unexpected message for a non-existed task")
        }
        case Some(content) => {
          val counter = content._2 - 1
          content._1.collect(piece)
          if (counter > 0) {
            taskLog += ((piece.parentId -> (content._1, counter)))
          } else {
            taskLog.remove(piece.parentId)
          }
        }
      }
    })
  }
}
