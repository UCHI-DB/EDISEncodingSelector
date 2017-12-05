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

import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

trait Evaluator {
  def init(): Unit

  def record(batch: Batch, loss: Double, acc: Int): Unit

  def loss: Double

  def accuracy: Double

  def summary: String
}

class MeanLossEvaluator extends Evaluator {
  var batchCounter = 0
  var itemCounter = 0
  var lossSum = 0d
  var accSum = 0

  def init(): Unit = {
    batchCounter = 0
    itemCounter = 0
    lossSum = 0
    accSum = 0
  }

  def record(batch: Batch, loss: Double, acc: Int) = {
    batchCounter += 1
    itemCounter += batch.size
    lossSum += loss
    accSum += acc
  }

  def loss: Double = lossSum / batchCounter

  def accuracy: Double = accSum.toDouble / itemCounter

  def summary: String =
    """Average loss %f, average accuracy %f""".format(loss, accuracy)
}

trait Trainer[T <: Dataset, G <: Graph] {
  val logger = LoggerFactory.getLogger(getClass)

  val trainHistory = new ArrayBuffer[(Double, Double, Double)]

  def getTrainSet: T

  def getTestSet: T

  def getGraph: G

  protected def getParamStore: ParamStore = EmptyStore

  protected def getEvaluator: Evaluator

  protected def earlyStop = false

  def train(epoches: Int, trainBatchSize: Int = 50, testBatchSize: Int = 50): Unit = {
    val trainset = getTrainSet

    val graph = getGraph
    val loadedParams = getParamStore.load
    if (loadedParams.nonEmpty)
      graph.load(loadedParams)

    // Initial test and loss
    var bestLoss = evaluate(testBatchSize)

    logger.info("Initial Test Result: %s".format(getEvaluator.summary))

    var i = 0
    var stop = false

    while (i < epoches && !stop) {
      logger.info("[Epoch %d]".format(i))
      val startTime = System.currentTimeMillis()

      val graph = getGraph

      getEvaluator.init()
      trainset.batches(trainBatchSize).foreach(batch => {
        setupGraph(graph, batch)
        val loss = graph.train
        getEvaluator.record(batch, loss, -1)
      })
      graph.epochDone()

      val trainLoss = getEvaluator.loss
      val testLoss = evaluate(testBatchSize)

      trainHistory += ((trainLoss, testLoss, getEvaluator.accuracy))

      if (testLoss < bestLoss) {
        bestLoss = testLoss
        // Only store the param when better result found
        if (null != getParamStore) {
          getParamStore.store(graph.dump())
        }
      }

      stop = earlyStop

      val stopTime = System.currentTimeMillis()

      logger.info("Training time %f mins".format((stopTime - startTime) / 60000d))
      logger.info("Training Loss %f, test loss %f".format(trainLoss, testLoss))
      logger.info("Test Result: %s".format(getEvaluator.summary))

      i += 1
    }
  }

  protected def evaluate(testBatchSize: Int): Double = {
    val testset = getTestSet
    val evaluator = getEvaluator
    val graph = getGraph

    evaluator.init()

    testset.batches(testBatchSize).foreach {
      batch => {
        setupGraph(graph, batch)
        val (loss, acc) = graph.test
        evaluator.record(batch, loss, acc)
      }
    }
    evaluator.loss
  }

  protected def setupGraph(graph: G, batch: Batch): Unit = {
    graph.build(batch)
  }
}

abstract class SimpleTrainer[T <: Dataset, G <: Graph]
(trainset: T, testset: T, graph: G, val modelName: String = "model")
  extends Trainer[T, G] {

  protected var evaluator: Evaluator = new MeanLossEvaluator()
  protected var paramStore: ParamStore = new FileStore(modelName)

  def getTrainSet: T = trainset

  def getTestSet: T = testset

  def getGraph: G = graph

  override protected def getParamStore = paramStore

  override protected def getEvaluator = evaluator
}
