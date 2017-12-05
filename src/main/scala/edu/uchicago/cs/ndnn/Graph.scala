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
package edu.uchicago.cs.ndnn

import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.GenIterable
import scala.collection.mutable.ArrayBuffer

class Graph(ip: InitPolicy, up: UpdatePolicy, loss: LossFunction) extends NodeEnv {

  val initPolicy = ip
  val updatePolicy = up
  val lossFunction = loss

  protected val inputs = new ArrayBuffer[Input]
  protected val params = new ArrayBuffer[Param]
  protected var expected: INDArray = _
  protected var out: Node = _

  def param(n: String, shape: Array[Int])(implicit usePolicy: InitPolicy = initPolicy): Param = {
    val newparam = super.param(n)
    newparam.value = usePolicy.init(shape)
    params += newparam
    newparam
  }

  override def input(n: String): Input = {
    val newinput = super.input(n)
    inputs += newinput
    newinput
  }

  def expect(value: INDArray) = expected = value

  def output(node: Node): Unit = out = node

  def getInputs = inputs.clone()

  def getParams = params.clone()

  def getOutput = out

  override def reset(): Unit = {
    if (this.snapshotCount == -1)
      return
    nodeBuffer.slice(this.snapshotCount, nodeBuffer.length).foreach(item => {
      item match {
        case param: Param => params -= param
        case input: Input => inputs -= input
        case _ => Unit
      }
    })
    super.reset()
  }

  /**
    * Subclasses should override this method to setup the graph
    *
    * @param batch
    * @return
    */
  def build(batch: Batch): Unit = Unit

  def train: Double = {
    // Check environment
    if (updatePolicy == null)
      throw new IllegalArgumentException("Update Policy is missing")
    if (out == null)
      throw new IllegalArgumentException("No output nodes")
    if (lossFunction == null)
      throw new IllegalArgumentException("No loss function defined")

    forward()
    // Compute Loss
    val loss = lossFunction.loss(out.value, expected)

    // Backward
    out.grad = lossFunction.gradient
    backward()
    // Update Parameters
    params.foreach {
      updatePolicy.update
    }
    loss
  }

  def epochDone() = {
    updatePolicy.weightDecay()
  }

  def test: (Double, Int) = {
    if (out == null)
      throw new IllegalArgumentException("No output nodes")
    if (lossFunction == null)
      throw new IllegalArgumentException("No loss function defined")

    forward()
    // Compute Loss
    val loss = lossFunction.loss(out.value, expected, true)
    (loss, lossFunction.accuracy)
  }

  def dump(): Array[INDArray] = this.params.map(_.value).toArray

  def load(params: GenIterable[INDArray]) = this.params.zip(params).foreach(p => p._1.value = p._2)
}