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

/**
  * Created by harper on 5/3/17.
  */
trait Task {

  def id: String

  def status: TaskStatus

  def spawn(numPiece: Int): Seq[TaskPiece]

  def collect(piece: TaskPiece): Unit
}

trait TaskPiece extends Serializable {

  def parentId: String

  def id: String

  def parameter: java.io.Serializable

  def failed: Boolean

  def result: java.io.Serializable

  def execute()

  def markFail()
}

abstract class DefaultTask extends Task {

  val id: String = UUID.randomUUID().toString

  var status: TaskStatus = TaskStatus.SUCCESS

  def collect(piece: TaskPiece): Unit = {
    if (piece.failed) {
      status = TaskStatus.FAILURE
    }
  }
}

abstract class DefaultTaskPiece(parent: Task) extends TaskPiece {

  val parentId = parent.id

  val id = UUID.randomUUID().toString

  var parameter: java.io.Serializable = null

  var result: java.io.Serializable = null

  var failed: Boolean = false

  def markFail(): Unit = {
    failed = true
  }

}
