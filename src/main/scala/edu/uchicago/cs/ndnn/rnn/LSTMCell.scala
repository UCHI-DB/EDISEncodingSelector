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
package edu.uchicago.cs.ndnn.rnn

import edu.uchicago.cs.ndnn._

object LSTMCell {

  def build(wf: Param, bf: Param, wi: Param, bi: Param, wc: Param, bc: Param,
            wo: Param, bo: Param, in: Node, h: Node, c: Node): (Node, Node) = {
    val concat = new Concat(h, in)
    val fgate = new Sigmoid(new Add(new DotMul(concat, wf), bf))
    val igate = new Sigmoid(new Add(new DotMul(concat, wi), bi))
    val cgate = new Mul(new Tanh(new Add(new DotMul(concat, wc), bc)), igate)
    val ogate = new Sigmoid(new Add(new DotMul(concat, wo), bo))

    val co = new Add(new Mul(c, fgate), cgate)
    val ho = new Mul(new Tanh(co), ogate)
    (ho, co)
  }
}