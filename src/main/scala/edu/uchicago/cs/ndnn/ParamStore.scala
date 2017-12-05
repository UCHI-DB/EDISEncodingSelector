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

import java.io.{ DataInputStream, DataOutputStream, FileInputStream, FileOutputStream }

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.GenIterable
import scala.collection.mutable.ArrayBuffer
import java.nio.file.Files
import java.nio.file.Paths

/**
 * Interface to store and load params
 */
trait ParamStore {
  def load: GenIterable[INDArray]

  def store(params: Array[INDArray])
}

object EmptyStore extends ParamStore {
  def load: GenIterable[INDArray] = ArrayBuffer.empty[INDArray]
  def store(params: Array[INDArray]) = Unit
}

class FileStore(file: String) extends ParamStore {

  override def load = {
    if (Files.exists(Paths.get(file))) {
      val ins = new DataInputStream(new FileInputStream(file))
      val params = new ArrayBuffer[INDArray]
      while (ins.available() > 0)
        params += Nd4j.read(ins)
      ins.close()
      params
    } else {
      Array.empty[INDArray]
    }
  }

  override def store(params: Array[INDArray]) = {
    val os = new DataOutputStream(new FileOutputStream(file))
    params.foreach(param => Nd4j.write(param, os))
    os.close()
  }
}
