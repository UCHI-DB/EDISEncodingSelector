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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  *
  */
trait LossFunction {
  protected var grad: INDArray = _
  protected var acc: Int = -1

  def forClassification: Boolean

  /**
    * @return average loss within a given batch
    */
  def loss(actual: INDArray, expected: INDArray, fortest: Boolean = false): Double

  /**
    * @return the gradients correspond to each input. The gradient return here should
    *         be averaged on each batch, thus the backprop result can be directly sum
    *         in a batch
    */
  def gradient: INDArray = grad

  def accuracy: Int = acc
}

class SquareLoss extends LossFunction {

  def forClassification = false

  /**
    * @param actual     Shape [B, N]
    * @param 	expected Shape [B, N]
    * @return The averaged squared difference of actual - expected
    */
  def loss(actual: INDArray, expected: INDArray, fortest: Boolean): Double = {
    val b = actual.shape()(0)

    if (!fortest) {
      // Compute Gradient
      grad = actual.sub(expected).mul(1d / b)
    }
    // Loss
    Transforms.pow(actual.sub(expected), 2).sum(1).meanNumber().doubleValue() / 2
  }
}

object SoftMaxLogLoss {
  val clip = 1e-12
}

class LogLoss extends LossFunction {
  def forClassification = true

  /**
    *
    * @param actual
    * @param expect
    * @param fortest
    * @return average loss within a given batch
    */
  def loss(actual: INDArray, expect: INDArray, fortest: Boolean): Double = {
    val shape = actual.shape()

    val clipval = Transforms.max(actual, SoftMaxLogLoss.clip, false)

    val nexpect = Nd4j.onesLike(expect).subi(expect)
    val nactual = Nd4j.onesLike(actual).subi(actual)

    if (!fortest) {
      // Compute gradient
      val grad = Nd4j.zerosLike(actual)
      // -c/x + (1-c)/(1-x)
      this.grad =
        nexpect.div(nactual).sub(expect.div(actual)).div(shape.product)
    }
    // Accuracy for classification
    val predict = actual.gte(0.5)
    val eq = predict.eq(expect)
    acc = eq.sumNumber().intValue()

    // - c log(x) - (1-c)log(1-x)
    Transforms.log(actual.mul(expect).add(nactual.mul(nexpect))).negi()
      .meanNumber().doubleValue()
  }
}

class SoftMaxLogLoss extends LossFunction {

  def forClassification = true

  /**
    * @param 	actual   Probability of each label. This is an <code>INDArray</code>
    *                   of shape [B,..., N], where B is the batch size, N is the dimension
    * @param 	expected The ground truth label. This is an <code>INDArray</code> of
    *                   shape [B,..., 1]
    * @return mean of log loss of ground truth label in actual probability
    *
    */
  def loss(actual: INDArray, expected: INDArray, fortest: Boolean): Double = {
    val shape = actual.shape()

    val fetch = Index.get(actual, expected)
    val clipval = Transforms.max(fetch, SoftMaxLogLoss.clip, false)
    if (!fortest) {
      // Compute gradient
      val grad = Nd4j.zerosLike(actual)
      // -log(x) gradient
      val allone = Nd4j.ones(expected.shape(): _*).div(expected.shape.product)

      Index.put(grad, expected, allone.divi(clipval).negi())
      this.grad = grad
    }
    // Accuracy for classification
    val predict = Nd4j.argMax(actual, actual.shape.length - 1)
    val eq = predict.eqi(expected)
    acc = eq.sumNumber().intValue()

    Transforms.log(clipval, false).negi().meanNumber().doubleValue()
  }
}