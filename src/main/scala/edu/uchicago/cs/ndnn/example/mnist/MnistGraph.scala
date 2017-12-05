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
package edu.uchicago.cs.ndnn.example.mnist

import edu.uchicago.cs.ndnn._

class MnistGraph extends Graph(Xavier, new SGD(0.5, 1), new SoftMaxLogLoss) {

  val pixelInput = input("pixel")

  {
    val w1 = param("w1", Array(28 * 28, 128))
    val b1 = param("b1", Array(1, 128))(Zero)
    val w2 = param("w2", Array(128, 10))
    val b2 = param("b2", Array(1, 10))(Zero)

    val wx = new DotMul(pixelInput, w1)
    val addb = new Add(wx, b1)
    val relu = new Sigmoid(addb)
    val layer2 = new DotMul(relu, w2)
    val addb2 = new Add(layer2, b2)
    val softmax = new SoftMax(addb2)
    output(softmax)
  }
}