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

import java.util.Collections

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait Data {

  def length(): String

  def numPiece(): Int

  def feature(): Array[Array[Double]]

  def groundTruth(): Double
}

class DefaultData(val data: Array[Double], val gt: Double) extends Data {

  def length() = data.length.toString

  def numPiece(): Int = 1

  def feature(): Array[Array[Double]] = Array(data)

  def groundTruth(): Double = gt
}

trait Dataset {

  def batches(batchSize: Int): Iterator[Batch]

  def numBatch: Int

  def dataSize: Int
}

class Batch(val size: Int, val data: Array[Data]) {

  def feature(piece: Int): INDArray = {
    val sample = data(0)
    if (piece >= sample.numPiece())
      throw new IllegalArgumentException("%d: invalid index".format(piece))
    Nd4j.create(data.map(_.feature()(piece)))
  }

  def groundTruth(): INDArray = Nd4j.create(data.map(_.groundTruth())).reshape(size, -1)
}

abstract class DefaultDataset extends Dataset {

  protected val logger = LoggerFactory.getLogger(getClass())

  protected var datas: Array[Data] = _
  protected var permuteIdx: mutable.Buffer[Int] = _
  var numBatch = 0

  {
    datas = load()
    permuteIdx = datas.indices.toBuffer
  }

  def dataSize = datas.length

  protected def load(): Array[Data]

  def batches(batchSize: Int): Iterator[Batch] = {
    Collections.shuffle(permuteIdx)

    if (batchSize <= 0)
      throw new IllegalArgumentException("Batch Size should be non-negative")
    val batches = (datas.indices by batchSize)
    numBatch = batches.length

    batches.toIterator.map(i => {
      val idices = permuteIdx.slice(i, i + batchSize)
      construct(idices)
    })
  }

  protected def construct(idices: Seq[Int]): Batch = {
    new Batch(idices.length, idices.map(datas(_)).toArray)
  }

  /**
    * Generate a new dataset by fetching some records from current dataset.
    * Used for creating training / dev / test set
    *
    * @param ratio
    * @return
    */
  def split(ratio: Seq[Double]): Seq[Dataset] = {

    val length = this.datas.length
    val permutation = (0 until length).toBuffer
    Collections.shuffle(permutation)

    val slice = ratio.map(d => Math.floor(d * length).toInt).toBuffer
    val addup = length - slice.sum

    slice(slice.length - 1) += addup

    var pointer = 0

    slice.map(cnt => {
      val data = new ArrayBuffer[Data]
      (pointer until pointer + cnt).foreach(idx => {
        data += this.datas(permutation(idx))
      })
      pointer += cnt
      new PreparedDataset(data.toArray)
    })
  }
}

class PreparedDataset(val data: Array[Data])
  extends DefaultDataset {

  override def load(): (Array[Data]) = data
}


abstract class VarLenDataset extends Dataset {

  protected var datas: Array[Data] = _
  protected var groupByLength: Array[Array[Int]] = _
  protected var permuteIdx: mutable.Buffer[Int] = _

  var numBatch = 0

  {
    datas = load()
    groupByLength = datas.zipWithIndex.map(d => (d._1.length(), d._2)).groupBy(_._1)
      .values.map(_.map(_._2)).toArray
    permuteIdx = groupByLength.indices.toBuffer
  }

  def dataSize = datas.length

  protected def load(): Array[Data]

  def batches(batchSize: Int): Iterator[Batch] = {
    Collections.shuffle(permuteIdx)

    if (batchSize <= 0)
      throw new IllegalArgumentException("Batch Size should be non-negative")

    numBatch = groupByLength.map(group => (group.indices by batchSize).length).sum

    permuteIdx.flatMap(idx => {
      val group = groupByLength(idx)
      (0 until group.length by batchSize).map(iidx => {
        construct(group.slice(iidx, iidx + batchSize))
      })
    }).toIterator
  }

  protected def construct(idices: Seq[Int]): Batch = {
    new Batch(idices.length, idices.map(datas(_)).toArray)
  }
}

