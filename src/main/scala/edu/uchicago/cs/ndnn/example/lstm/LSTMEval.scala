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
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

/**
  * Created by harper on 6/2/17.
  */
object LSTMEval extends App {
  val dict = new LSTMDict()
  Source.fromFile("/home/harper/dataset/bobsue-data/bobsue.voc.txt").getLines().foreach(dict.update)

  val trainds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.train.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)
  val devds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.dev.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)
  val testds = new S2SDataset("/home/harper/dataset/bobsue-data/bobsue.seq2seq.test.tsv",
    new WordTokenizer(), "<s>", "</s>", dict, dict)

  val graph = new LSTMEncodeGraph(dict.size, dict.size, 200, new Adam(0.01, 0.9, 0.999))

  graph.enc_embed.set(Nd4j.readNumpy("temp_model/enc_embed.npy", "\t"))
  graph.enc_v2c.set(Nd4j.readNumpy("temp_model/enc_v2c.npy", "\t"))
  graph.enc_wc.set(Nd4j.readNumpy("temp_model/enc_wc.npy", "\t"))
  graph.enc_bc.set(Nd4j.readNumpy("temp_model/enc_bc.npy", "\t"))
  graph.enc_wf.set(Nd4j.readNumpy("temp_model/enc_wf.npy", "\t"))
  graph.enc_bf.set(Nd4j.readNumpy("temp_model/enc_bf.npy", "\t"))
  graph.enc_wo.set(Nd4j.readNumpy("temp_model/enc_wo.npy", "\t"))
  graph.enc_bo.set(Nd4j.readNumpy("temp_model/enc_bo.npy", "\t"))
  graph.enc_wi.set(Nd4j.readNumpy("temp_model/enc_wi.npy", "\t"))
  graph.enc_bi.set(Nd4j.readNumpy("temp_model/enc_bi.npy", "\t"))

  graph.dec_embed.set(Nd4j.readNumpy("temp_model/dec_embed.npy", "\t"))
  graph.dec_v2c.set(Nd4j.readNumpy("temp_model/dec_v2c.npy", "\t"))
  graph.dec_wc.set(Nd4j.readNumpy("temp_model/dec_wc.npy", "\t"))
  graph.dec_bc.set(Nd4j.readNumpy("temp_model/dec_bc.npy", "\t"))
  graph.dec_wf.set(Nd4j.readNumpy("temp_model/dec_wf.npy", "\t"))
  graph.dec_bf.set(Nd4j.readNumpy("temp_model/dec_bf.npy", "\t"))
  graph.dec_wo.set(Nd4j.readNumpy("temp_model/dec_wo.npy", "\t"))
  graph.dec_bo.set(Nd4j.readNumpy("temp_model/dec_bo.npy", "\t"))
  graph.dec_wi.set(Nd4j.readNumpy("temp_model/dec_wi.npy", "\t"))
  graph.dec_bi.set(Nd4j.readNumpy("temp_model/dec_bi.npy", "\t"))

  val eval = new LSTMEvaluator()
  eval.init()
  testds.batches(50).foreach(batch => {
    graph.build(batch)
    val (loss, acc) = graph.test
    eval.record(batch, loss, acc)
  })
  println(eval.accuracy)
}
