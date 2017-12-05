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

package edu.uchicago.cs.ndnn.example.lstm

import edu.uchicago.cs.ndnn.Adam
import edu.uchicago.cs.ndnn.rnn.{LSTMDict, LSTMEncodeGraph, S2SDataset, WordTokenizer}

import scala.io.Source

/**
  * Created by harper on 6/1/17.
  */
object LSTM extends App {

  val dict = new LSTMDict()
  Source.fromFile("/home/harper/dataset/bobsue-data/bobsue.voc.txt").getLines().foreach(dict.update)

  val trainds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.train.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)
  val devds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.dev.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)
  val testds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.test.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)

  val graph = new LSTMEncodeGraph(dict.size, dict.size, 200, new Adam(0.01, 0.9, 0.999))

  val trainer = new LSTMTrainer(trainds, testds, graph, "lstm_encoder")
  trainer.train(30)
}

