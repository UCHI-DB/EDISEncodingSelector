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

import java.io.DataInputStream
import java.io.FileInputStream

import edu.uchicago.cs.ndnn.{Data, DefaultData, DefaultDataset}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ArrayBuffer

object MinstDataset {
  val DATA_MAGIC = 2051
  val LABEL_MAGIC = 2049
}

class MnistDataset(trainFile: String, testFile: String, sizeLimit: Int = -1) extends DefaultDataset {

  def load(): Array[Data] = {
    val datais = new DataInputStream(new FileInputStream(trainFile))
    val labelis = new DataInputStream(new FileInputStream(testFile))
    // Magic number
    val datamagic = datais.readInt()
    val labelmagic = labelis.readInt()
    if (MinstDataset.DATA_MAGIC != datamagic || MinstDataset.LABEL_MAGIC != labelmagic)
      throw new IllegalArgumentException("Incorrect magic number: %d, %d".format(datamagic, labelmagic))

    val datacount = datais.readInt()
    val labelcount = labelis.readInt()
    if (datacount != labelcount)
      throw new IllegalArgumentException("Incorrect number of records")

    val dataSize = sizeLimit match {
      case -1 => datacount
      case _ => sizeLimit
    }

    val rowcnt = datais.readInt()
    val colcnt = datais.readInt()
    if (rowcnt != 28 || colcnt != 28)
      throw new IllegalArgumentException("Incorrect row/col cnt")

    val data = new ArrayBuffer[Data]

    for (i <- 0 until dataSize) {
      val databuffer = new Array[Double](rowcnt * colcnt)
      for (j <- databuffer.indices)
        databuffer(j) = datais.readUnsignedByte() / 255.toDouble
      data += new DefaultData(databuffer, labelis.readUnsignedByte().toDouble)
    }

    datais.close()
    labelis.close()

    data.toArray
  }
}