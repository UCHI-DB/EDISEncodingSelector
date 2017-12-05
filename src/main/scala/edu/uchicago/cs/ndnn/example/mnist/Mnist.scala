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

import java.io.File
import java.nio.file.{Files, Paths}

object Mnist extends App {
  val folder = "/home/harper"

  val trainDataFile = folder + "/dataset/mnist/train-images.idx3-ubyte"
  val trainLabelFile = folder + "/dataset/mnist/train-labels.idx1-ubyte"
  val testDataFile = folder + "/dataset/mnist/t10k-images.idx3-ubyte"
  val testLabelFile = folder + "/dataset/mnist/t10k-labels.idx1-ubyte"

  if (!Files.exists(Paths.get(new File(trainDataFile).toURI)))
    throw new IllegalArgumentException("Cannot find dataset files")

  val trainset = new MnistDataset(trainDataFile, trainLabelFile)
  val testset = new MnistDataset(testDataFile, testLabelFile)

  val trainer = new MnistTrainer(trainset, testset)
  trainer.train(30)
}