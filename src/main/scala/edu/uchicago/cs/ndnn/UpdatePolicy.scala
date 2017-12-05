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

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object UpdatePolicy {
  val etaDefault = 0.5
  val etaDecay = 1

  val gradClip = -1

  val momentumKey = "momentum"

  val rmspropKey = "rmsprop"
  val rmsEpsilon = 1e-8

  val adammeanKey = "adammean"
  val adamvarKey = "adamvar"

  val adamAlpha = 0.9
  val adamBeta = 0.999


  val dsdmaskKey = "dsdmask"
  val dsdthreshold = 0.001
}

trait UpdatePolicy {
  def update(p: Param): Unit

  def weightDecay(): Unit
}

abstract class UpdatePolicyBase(e: Double, d: Double, gc: Double) extends UpdatePolicy {
  protected var eta = e
  protected val decay = d
  val gradClip = gc

  def update(p: Param): Unit = {
    clipGrad(p)
    innerUpdate(p)
  }

  def weightDecay() = eta *= decay

  protected def clipGrad(p: Param) = {
    if (gradClip > 0) {
      val norm = p.grad.norm2Number().doubleValue()
      if (norm >= gradClip)
        p.grad.muli(gradClip / norm)
    }
  }

  protected def innerUpdate(p: Param): Unit
}

class SGD(e: Double, d: Double, gc: Double) extends UpdatePolicyBase(e, d, gc) {

  def this() = this(UpdatePolicy.etaDefault, UpdatePolicy.etaDecay, UpdatePolicy.gradClip)

  def this(e: Double) = this(e, UpdatePolicy.etaDecay, UpdatePolicy.gradClip)

  def this(e: Double, d: Double) = this(e, d, UpdatePolicy.gradClip)

  def innerUpdate(p: Param) = {
    p.value.subi(p.grad.mul(eta))
  }

}

class Momentum(e: Double, d: Double, m: Double, gc: Double) extends UpdatePolicyBase(e, d, gc) {

  private val mu = m

  def this(e: Double, m: Double) = this(e, UpdatePolicy.etaDecay, m, UpdatePolicy.gradClip)

  def this(m: Double) = this(UpdatePolicy.etaDefault, UpdatePolicy.etaDecay, m, UpdatePolicy.gradClip)

  def innerUpdate(p: Param) = {
    val grad = p.grad
    val oldmomen = p.context.getOrElse(UpdatePolicy.momentumKey, Nd4j.zerosLike(p.grad))
    val momentum = oldmomen.mul(mu).add(grad)
    p.value.subi(momentum.mul(eta))
    p.context.put(UpdatePolicy.momentumKey, momentum)
  }

}

class RMSProp(e: Double, d: Double, b: Double, gc: Double) extends UpdatePolicyBase(e, d, gc) {

  private val beta = b

  def this(e: Double, b: Double) = this(e, UpdatePolicy.etaDecay, b, UpdatePolicy.gradClip)

  def this(b: Double) = this(UpdatePolicy.etaDefault, UpdatePolicy.etaDecay, b, UpdatePolicy.gradClip)

  def innerUpdate(p: Param) = {
    val grad = p.grad
    val gradsqr = Transforms.pow(grad, 2)
    val oldrms = p.context.getOrElse(UpdatePolicy.rmspropKey, Nd4j.zerosLike(p.grad))
    val rms = oldrms.mul(beta).add(gradsqr.mul(1 - beta))
    p.value.subi(grad.mul(eta).div(Transforms.sqrt(rms.add(UpdatePolicy.rmsEpsilon))))
    p.context.put(UpdatePolicy.rmspropKey, rms)
  }

}

class Adam(e: Double, d: Double, a: Double, b: Double, gc: Double) extends UpdatePolicyBase(e, d, gc) {

  private val alpha = a
  private val beta = b

  def this(e: Double, a: Double, b: Double) = this(e, UpdatePolicy.etaDecay, a, b, UpdatePolicy.gradClip)

  def this(a: Double, b: Double) = this(UpdatePolicy.etaDefault, UpdatePolicy.etaDecay, a, b, UpdatePolicy.gradClip)

  def this() = this(UpdatePolicy.adamAlpha, UpdatePolicy.adamBeta)

  def innerUpdate(p: Param) = {
    val grad = p.grad
    val gradsqr = Transforms.pow(grad, 2)
    val oldmomen = p.context.getOrElse(UpdatePolicy.adammeanKey, Nd4j.zerosLike(grad))
    val momentum = oldmomen.mul(alpha).add(grad.mul(1 - alpha))

    val oldrms = p.context.getOrElse(UpdatePolicy.adamvarKey, Nd4j.zerosLike(grad))
    val rms = oldrms.mul(beta).add(gradsqr.mul(1 - beta))
    p.value.subi(momentum.mul(eta).div(Transforms.sqrt(rms.add(UpdatePolicy.rmsEpsilon))))

    p.context.put(UpdatePolicy.adammeanKey, momentum)
    p.context.put(UpdatePolicy.adamvarKey, rms)
  }
}

class DenseSparseDense(child: UpdatePolicy, period1: Int, period2: Int,
                       threshold: Double = UpdatePolicy.dsdthreshold) extends UpdatePolicy {

  private var epochCounter = 0

  def update(p: Param): Unit = {
    child.update(p)
    if (period1 <= epochCounter && epochCounter < period2) {
      val mask = p.context.getOrElseUpdate(UpdatePolicy.dsdmaskKey, {
        Transforms.abs(p.value).gt(threshold)
      })
      p.value.muli(mask)
    }
  }

  def weightDecay(): Unit = {
    child.weightDecay()
    epochCounter += 1
  }
}