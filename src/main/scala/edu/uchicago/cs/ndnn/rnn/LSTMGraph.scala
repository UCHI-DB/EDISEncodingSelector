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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import scala.collection.mutable.ArrayBuffer

class LSTMGraph(val dictSize: Int, val hiddenDim: Int, updatePolicy: UpdatePolicy = new Adam())
  extends Graph(Xavier, updatePolicy, new SoftMaxLogLoss) {


  val c2v = param("c2v", Array(dictSize, hiddenDim))
  val v2c = param("v2c", Array(hiddenDim, dictSize))
  val inputSize = 2 * hiddenDim

  val wf = param("wf", Array(inputSize, hiddenDim))
  val bf = param("bf", Array(1, hiddenDim))(Zero)
  val wi = param("wi", Array(inputSize, hiddenDim))
  val bi = param("bi", Array(1, hiddenDim))(Zero)
  val wc = param("wc", Array(inputSize, hiddenDim))
  val bc = param("bc", Array(1, hiddenDim))(Zero)
  val wo = param("wo", Array(inputSize, hiddenDim))
  val bo = param("bo", Array(1, hiddenDim))(Zero)

  val h0 = input("h0")
  val c0 = input("c0")

  val xs = new ArrayBuffer[Input]()

  snapshot()

  override def build(batch: Batch): Unit = {
    reset()
    // Shape B,L,H
    val data = batch.feature(0)
    val shape = data.shape()
    val batchSize = shape(0)
    val length = shape(1)
    if (length <= 1)
      throw new IllegalArgumentException()

    h0.set(Nd4j.zeros(batchSize, hiddenDim))
    c0.set(Nd4j.zeros(batchSize, hiddenDim))

    val collected = new ArrayBuffer[Node]()
    // Extend RNN to the expected size and build connections between cells
    var h: Node = h0
    var c: Node = c0
    for (i <- 0 until length - 1) {
      val in = input("%d".format(i))
      in.set(data.get(NDArrayIndex.all(), NDArrayIndex.point(i)))
      val mapped = new Embed(in, c2v)
      xs += in

      // H,C
      val newNode = LSTMCell.build(wf, bf, wi, bi,
        wc, bc, wo, bo, mapped, h, c)
      h = newNode._1
      c = newNode._2
      collected += new SoftMax(new DotMul(h, v2c))
    }
    output(new Collect(collected: _*))
    expect(data.get(NDArrayIndex.all(), NDArrayIndex.interval(1, length), NDArrayIndex.newAxis()))
  }
}

class LSTMEncodeGraph(val encDictSize: Int, val decDictSize: Int,
                      val hiddenDim: Int, updatePolicy: UpdatePolicy = new Adam())
  extends Graph(Xavier, updatePolicy, new SoftMaxLogLoss) {

  val inputSize = 2 * hiddenDim

  val enc_embed = param("enc_embed", Array(encDictSize, hiddenDim))

  val enc_wf = param("enc_wf", Array(inputSize, hiddenDim))
  val enc_bf = param("enc_bf", Array(1, hiddenDim))(Zero)
  val enc_wi = param("enc_wi", Array(inputSize, hiddenDim))
  val enc_bi = param("enc_bi", Array(1, hiddenDim))(Zero)
  val enc_wc = param("enc_wc", Array(inputSize, hiddenDim))
  val enc_bc = param("enc_bc", Array(1, hiddenDim))(Zero)
  val enc_wo = param("enc_wo", Array(inputSize, hiddenDim))
  val enc_bo = param("enc_bo", Array(1, hiddenDim))(Zero)

  val enc_v2c = param("enc_v2c", Array(hiddenDim, encDictSize))

  val enc_h0 = input("enc_h0")
  val enc_c0 = input("enc_c0")


  val dec_embed = param("dec_embed", Array(decDictSize, hiddenDim))

  val dec_wf = param("dec_wf", Array(inputSize, hiddenDim))
  val dec_bf = param("dec_bf", Array(1, hiddenDim))(Zero)
  val dec_wi = param("dec_wi", Array(inputSize, hiddenDim))
  val dec_bi = param("dec_bi", Array(1, hiddenDim))(Zero)
  val dec_wc = param("dec_wc", Array(inputSize, hiddenDim))
  val dec_bc = param("dec_bc", Array(1, hiddenDim))(Zero)
  val dec_wo = param("dec_wo", Array(inputSize, hiddenDim))
  val dec_bo = param("dec_bo", Array(1, hiddenDim))(Zero)

  val dec_v2c = param("dec_v2c", Array(hiddenDim, decDictSize))

  val dec_h0 = input("dec_h0")
  val dec_c0 = input("dec_c0")

  snapshot()

  override def build(batch: Batch): Unit = {
    reset()
    val encData = batch.feature(0)
    val decData = batch.feature(1)

    val batchSize = encData.shape()(0)
    val encLength = encData.shape()(1)
    val decLength = decData.shape()(1)

    enc_h0.set(Nd4j.zeros(batchSize, hiddenDim))
    enc_c0.set(Nd4j.zeros(batchSize, hiddenDim))

    var h: Node = enc_h0
    var c: Node = enc_c0

    for (i <- 0 until encLength) {
      val in_i = input("enc_in_%d".format(i))
      in_i.set(encData.get(Index.point(2, 1, i): _*).reshape(batchSize, 1))
      val embed = new Embed(in_i, enc_embed)

      val (th, tc) = LSTMCell.build(enc_wf, enc_bf, enc_wi, enc_bi,
        enc_wc, enc_bc, enc_wo, enc_bo, embed, h, c)
      h = th
      c = tc
    }

    val outputs = new ArrayBuffer[Node]()
    for (i <- 0 until decLength - 1) {
      val in_i = input("dec_in_%d".format(i))
      in_i.set(decData.get(Index.point(2, 1, i): _*).reshape(batchSize, 1))
      val embed = new Embed(in_i, dec_embed)
      val (th, tc) = LSTMCell.build(dec_wf, dec_bf, dec_wi, dec_bi,
        dec_wc, dec_bc, dec_wo, dec_bo, embed, h, c)
      h = th
      c = tc
      outputs += new SoftMax(new DotMul(h, dec_v2c))
    }

    output(new Collect(outputs: _*))
    expect(decData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, decLength)))
  }
}

class LSTMDecodeGraph(encDictSize: Int, decDictSize: Int, hiddenDim: Int,
                      val decodeLength: Int, val decStartSymbol: Int)
  extends LSTMEncodeGraph(encDictSize, decDictSize, hiddenDim, null) {

  override def build(batch: Batch): Unit = {
    reset()
    val encData = batch.feature(0)

    val batchSize = encData.shape()(0)
    val encLength = encData.shape()(1)

    enc_h0.set(Nd4j.zeros(batchSize, hiddenDim))
    enc_c0.set(Nd4j.zeros(batchSize, hiddenDim))

    var h: Node = enc_h0
    var c: Node = enc_c0

    for (i <- 0 until encLength) {
      val in_i = input("enc_in_%".format(i))
      in_i.set(encData.get(Index.point(2, 1, i): _*).reshape(batchSize, 1))
      val embed = new Embed(in_i, enc_embed)
      val (th, tc) = LSTMCell.build(enc_wf, enc_bf, enc_wi, enc_bi,
        enc_wc, enc_bc, enc_wo, enc_bo, embed, h, c)
      h = th
      c = tc
    }

    val dec_init_input = input("dec_in_start")
    dec_init_input.set(Nd4j.createUninitialized(Array(batchSize, 1)).assign(decStartSymbol))

    var dec_input: Node = dec_init_input

    val outputs = new ArrayBuffer[Node]
    for (i <- 0 until decodeLength) {
      val embed = new Embed(dec_input, dec_embed)
      val (th, tc) = LSTMCell.build(dec_wf, dec_bf, dec_wi, dec_bi,
        dec_wc, dec_bc, dec_wo, dec_bo, embed, h, c)
      h = th
      c = tc
      dec_input = new ArgMax(new SoftMax(new DotMul(h, dec_v2c)))
      outputs += dec_input
    }
    output(new Collect(outputs: _*))
  }
}

