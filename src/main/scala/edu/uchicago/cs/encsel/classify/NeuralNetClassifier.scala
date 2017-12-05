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

package edu.uchicago.cs.encsel.classify.nn

import edu.uchicago.cs.encsel.classify.{EncSelNNGraph, EncselDataset}
import edu.uchicago.cs.encsel.model.DataType
import edu.uchicago.cs.ndnn._
//import edu.uchicago.cs.ndnn.figure.LineFigure

import scala.collection.mutable.ArrayBuffer

/**
  * Use Neural Network to choose encoding
  */
object ClassifierForInt extends App {
  val prefix = args.length match {
    case 0 => ""
    case _ => args(0)
  }
  val fullds = new EncselDataset(DataType.INTEGER, prefix)
  val datasets = fullds.split(Seq(0.9, 0.1))
  val trainds = datasets.head
  val testds = datasets(1)

  val graph = new EncSelNNGraph(fullds.numFeature, fullds.numClass)
  val trainer = new SimpleTrainer[Dataset, EncSelNNGraph](trainds, testds,
    graph, "enc_nn_int_%smodel".format(prefix)) {

    override def setupGraph(graph: EncSelNNGraph, batch: Batch): Unit = {
      graph.x.set(batch.feature(0))
      graph.expect(batch.groundTruth)
    }
  }
  trainer.train(200)

  val trainLoss = new ArrayBuffer[(Number, Number)]
  val testLoss = new ArrayBuffer[(Number, Number)]
  val accuracy = new ArrayBuffer[(Number, Number)]

  trainer.trainHistory.zipWithIndex.foreach(item => {
    trainLoss += ((item._2, item._1._1))
    testLoss += ((item._2, item._1._2))
    accuracy += ((item._2, item._1._3))
  })

//  new LineFigure().title("").xLabel("Epoch").yLabel("")
//    .addPlot(trainLoss, "Training Loss")
//    .addPlot(testLoss, "Test Loss")
//    .addPlot(accuracy, "Test Accuracy")
//    .show()
}

object ClassifierForString extends App {
  val prefix = args.length match {
    case 0 => ""
    case _ => args(0)
  }
  val fullds = new EncselDataset(DataType.STRING, prefix)
  val datasets = fullds.split(Seq(0.9, 0.1))
  val trainds = datasets.head
  val testds = datasets(1)

  val graph = new EncSelNNGraph(fullds.numFeature, fullds.numClass)
  val trainer = new SimpleTrainer[Dataset, EncSelNNGraph](trainds, testds, graph) {

    {
      paramStore = new FileStore("enc_nn_string_%smodel".format(prefix))
    }

    override def setupGraph(graph: EncSelNNGraph, batch: Batch): Unit = {
      graph.x.set(batch.data)
      graph.expect(batch.groundTruth)
    }
  }
  trainer.train(200)
}

object DataAnalysis extends App {

  val fullds = new EncselDataset(DataType.INTEGER) {
    def analysis: Map[Int, Int] = {
      this.datas.map(data => data.groundTruth()).groupBy(_.toInt).map(f => (f._1, f._2.length))
    }
  }

  val result = fullds.analysis
  println(result)
}
