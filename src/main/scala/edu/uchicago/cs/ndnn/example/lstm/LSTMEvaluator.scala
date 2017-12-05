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

package edu.uchicago.cs.ndnn.example.lstm

import edu.uchicago.cs.ndnn.{Batch, Evaluator}

/**
  * Created by harper on 6/1/17.
  */
class LSTMEvaluator extends Evaluator {

  var totalLoss = 0d
  var totalAcc = 0d
  var itemCounter = 0

  def init(): Unit = {
    itemCounter = 0
    totalLoss = 0
    totalAcc = 0
  }

  def record(batch: Batch, loss: Double, acc: Int): Unit = {
    itemCounter += batch.size * (batch.feature(1).shape()(1) - 1)
    totalLoss += loss
    totalAcc += acc
  }

  def accuracy: Double = {
    totalAcc / itemCounter
  }

  def loss: Double = {
    totalLoss / itemCounter
  }

  def summary: String = {
    """Average loss %f, average accuracy %f""".format(loss, accuracy)
  }
}
