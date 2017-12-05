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

package edu.uchicago.cs.encsel.dataset

import edu.uchicago.cs.dist.{DefaultTask, DefaultTaskPiece, TaskPiece}

/**
  * Distribute the feature running on cluster
  */
class DistRunFeature {


}

class DistFeatureTask extends DefaultTask {

  def spawn(numPiece: Int): Seq[TaskPiece] = {
    Seq()
  }
}

class DistFeatureTaskPiece(parent: DistFeatureTask) extends DefaultTaskPiece(parent) {
  override def execute(): Unit = {

  }
}
