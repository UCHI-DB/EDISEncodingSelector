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

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.SpecifiedIndex
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.util.NDArrayUtil
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLUDerivative

trait NodeEnv {
  protected val nodeBuffer = new ArrayBuffer[Node]

  def attach(n: Node) = {
    nodeBuffer += n
  }

  def forward(): Unit =
    nodeBuffer.foreach {
      _.forward()
    }

  def backward(): Unit =
    nodeBuffer.reverseIterator.foreach(_.backward())

  def param(n: String): Param = {
    val newparam = new Param(n)
    newparam.setEnv(this)
    newparam
  }

  def input(n: String): Input = {
    val newinput = new Input(n)
    newinput.setEnv(this)
    newinput
  }

  protected var snapshotCount = -1

  def snapshot(): Unit = {
    this.snapshotCount = nodeBuffer.size
  }

  def reset(): Unit = {
    nodeBuffer.remove(this.snapshotCount, nodeBuffer.length - this.snapshotCount)
  }
}

abstract class Node(is: Node*) {

  private[ndnn] var value: INDArray = _
  private[ndnn] var grad: INDArray = _

  private[ndnn] var env: NodeEnv = _

  if (is.nonEmpty) {
    attachTo(is(0).env)
  }

  protected def attachTo(e: NodeEnv): Unit = {
    e.attach(this)
    this.env = e
  }

  def forward(): Unit = {
    compute()
    if (this.value != null) {
      if (this.grad == null ||
        !this.grad.shape.sameElements(this.value.shape)) {
        this.grad = Nd4j.zerosLike(this.value)
      } else {
        this.grad.assign(0)
      }
    }
  }

  def backward(): Unit = updateGrad()

  def compute(): Unit

  def updateGrad(): Unit

  def getValue = value
}

class Input(n: String) extends Node {
  val name = n

  def this() = this("default_input")

  def setEnv(e: NodeEnv): Unit = {
    this.env = e
    e.attach(this)
  }

  protected var rawValue: Any = _

  def get[T] = this.rawValue.asInstanceOf[T]

  def set(value: Any) = {
    this.rawValue = value
    if (value.isInstanceOf[INDArray])
      this.value = value.asInstanceOf[INDArray]
    if (value.isInstanceOf[Array[Double]]) {
      this.value = Nd4j.create(value.asInstanceOf[Array[Double]])
    }
    if (value.isInstanceOf[Array[Int]]) {
      this.value = Nd4j.create(value.asInstanceOf[Array[Int]].map(_.toDouble))
    }
  }

  def compute(): Unit = Unit

  def updateGrad(): Unit = Unit
}

class Param(n: String) extends Input(n) {
  val context = new HashMap[String, INDArray]()
}

class Add(left: Node, right: Node) extends Node(left, right) {

  def compute(): Unit = {
    this.value = Broadcast.add(left.value, right.value)
  }

  def updateGrad(): Unit = {
    this.left.grad.addi(this.grad)

    this.right.grad.addi(right.value.isVector match {
      case false => this.grad
      case true =>
        this.grad.sum((0 until grad.rank - 1): _*)
    })
  }
}

class Mul(left: Node, right: Node) extends Node(left, right) {

  def compute(): Unit = {
    this.value = Broadcast.mul(left.value, right.value)
  }

  def updateGrad() = {
    right.grad.isVector match {
      case false => {
        left.grad.addi(right.value.mul(grad))
        right.grad.addi(left.value.mul(grad))
      }
      case true => {
        left.grad.addi(Broadcast.mul(grad, right.value))
        right.grad.addi(grad.mul(left.value).sum((0 until grad.rank() - 1): _*))
      }
    }
  }
}

class DotMul(left: Node, right: Node) extends Node(left, right) {

  def compute(): Unit = {
    this.value = left.value.mmul(right.value)
  }

  def updateGrad() = {
    left.grad.addi(grad.mmul(right.value.transpose()))
    right.grad.addi(left.value.transpose.mmul(grad))
  }
}

class ReLU(input: Node) extends Node(input) {
  def compute(): Unit = {
    this.value = Transforms.relu(input.value)
  }

  def updateGrad() = {
    input.grad.addi(this.value.gt(0).muli(this.grad))
  }
}

class LeakyReLU(input: Node) extends Node(input) {
  def compute(): Unit = {
    this.value = Transforms.leakyRelu(input.value)
  }

  def updateGrad() = {
    val op = new LeakyReLUDerivative(this.value)
    val derivative = Nd4j.getExecutioner.execAndReturn(op)
    input.grad.addi(derivative)
  }
}

class Sigmoid(input: Node) extends Node(input) {
  def compute(): Unit = {
    this.value = Transforms.sigmoid(input.value)
  }

  def updateGrad() = {
    val one = Nd4j.onesLike(this.value)
    input.grad.addi(one.subi(this.value).muli(this.value).muli(grad))
  }
}

class Tanh(input: Node) extends Node(input) {
  def compute(): Unit = {
    this.value = Transforms.tanh(input.value)
  }

  def updateGrad() = {
    val one = Nd4j.onesLike(this.value)
    input.grad.addi(this.value.sub(one.muli(value)).muli(grad))
  }
}

class SoftMax(input: Node) extends Node(input) {

  def compute(): Unit = {
    this.value = Operations.softmax(input.value.dup())
  }

  def updateGrad() = {
    val dup = this.value.dup()
    val gvdot = dup.muli(this.grad).sum(this.grad.shape.length - 1)

    val op = new BroadcastSubOp(this.grad, gvdot, this.grad.dup(),
      this.grad.shape.indices.dropRight(1): _*)
    input.grad.addi(Nd4j.getExecutioner.execAndReturn(op).muli(this.value))
  }
}

class Concat(left: Node, right: Node, idx: Int = 1) extends Node(left, right) {
  def compute(): Unit = {
    this.value = Nd4j.concat(idx, left.value, right.value)
  }

  def updateGrad() = {
    left.grad.addi(this.grad.get(NDArrayIndex.all(),
      NDArrayIndex.interval(0, left.value.shape()(1))))
    right.grad.addi(this.grad.get(NDArrayIndex.all(),
      NDArrayIndex.interval(left.value.shape()(1), this.value.shape()(1))))
  }
}

class Collect(nodes: Node*) extends Node(nodes: _*) {
  def compute(): Unit = {
    this.value = Nd4j.concat(1,
      nodes.map(_.value.get(NDArrayIndex.all(),
        NDArrayIndex.newAxis(),
        NDArrayIndex.all())): _*)
  }

  def updateGrad() = {
    nodes.zipWithIndex.foreach(n => {
      n._1.grad = this.grad.get(Index.point(3, 1, n._2): _*)
    })
  }
}

/**
  * [B,1] [C,N] => [B,N]
  */
class Embed(idx: Node, map: Node) extends Node(idx, map) {

  var fwdmap: INDArray = _
  var bwdmap: INDArray = _

  def compute(): Unit = {

    val indexval = NDArrayUtil.toInts(idx.value)
    val mapsize = map.value.shape()
    if (fwdmap == null || !fwdmap.shape().sameElements(Array(indexval, mapsize(0)))) {
      fwdmap = Nd4j.zeros(indexval.length, mapsize(0))
    } else {
      fwdmap.assign(0)
    }
    indexval.zipWithIndex.foreach(p => fwdmap.putScalar(Array(p._2, p._1), 1))
    bwdmap = fwdmap.transpose()

    this.value = fwdmap.mmul(map.value)
  }

  def updateGrad() = {
    // Support only column vectors
    map.grad.addi(bwdmap.mmul(this.grad))
  }
}

class ArgMax(input: Node) extends Node(input) {

  def compute(): Unit = {
    this.value = Nd4j.argMax(input.value, 1)
  }

  def updateGrad() = {
    // TODO This is jvm loop and has performance issue
    val zero = Nd4j.zerosLike(input.value)
    Index.put(zero, this.value, this.grad)
    input.grad.addi(zero)
  }
}
