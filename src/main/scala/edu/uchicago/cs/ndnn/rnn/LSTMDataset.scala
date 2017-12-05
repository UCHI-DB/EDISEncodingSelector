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

import scala.collection.mutable.{ArrayBuffer, Buffer, HashMap}
import scala.io.Source

/**
  * An instance of <code>LSTMTokenizer</code> split input string into
  * tokens
  */
trait LSTMTokenizer {
  def tokenize(input: String): Iterable[String]
}

class WordTokenizer extends LSTMTokenizer {

  override def tokenize(input: String): Iterable[String] = {
    return input.split("\\s+")
  }
}

class CharTokenizer extends LSTMTokenizer {

  override def tokenize(input: String): Iterable[String] = {
    return input.toCharArray.map(_.toString)
  }
}

class LSTMData(data: Array[Double]) extends Data {
  def length(): String = data.length.toString

  def numPiece(): Int = 1

  def feature(): Array[Array[Double]] = Array(data)

  def groundTruth(): Double = 0d
}

class S2SData(from: Array[Double], to: Array[Double]) extends Data {

  def length(): String = (from.length, to.length).toString()

  def numPiece(): Int = 2

  def feature(): Array[Array[Double]] = Array(from, to)

  def groundTruth(): Double = 0d
}

class LSTMDict {
  private val dict: HashMap[String, Int] = new HashMap[String, Int]
  private val inverseDict: Buffer[String] = new ArrayBuffer[String]

  def update(word: String): Int = {
    dict.getOrElseUpdate(word, {
      inverseDict += word
      dict.size
    })
  }

  def lookup(word: String): Int = {
    dict.getOrElse(word, -1)
  }

  def size = dict.size
}

class LSTMDataset(file: String, val tokenizer: LSTMTokenizer,
                  val bosSymbol: String, val eosSymbol: String,
                  extdict: LSTMDict = null) extends VarLenDataset {

  protected var dict: LSTMDict = _

  override protected def load(): Array[Data] = {
    dict = new LSTMDict()
    val source = Source.fromFile(file)
    try {
      val content = source.getLines().map(line => tokenizer.tokenize(line))
      if (extdict == null) {
        dict.update(bosSymbol)
        dict.update(eosSymbol)
        content.map(parse(_, true)).toArray
      } else {
        dict = extdict
        content.map(parse(_, false)).toArray
      }
    } finally {
      source.close()
    }
  }

  protected def parse(input: Iterable[String], append: Boolean): Data = {
    var data = input.toArray
    if (!bosSymbol.equals(data(0))) {
      data = bosSymbol +: data
    }
    if (!eosSymbol.equals(data.last)) {
      data = data :+ eosSymbol
    }
    new LSTMData(append match {
      case true => {
        data.map(dict.update(_).toDouble)
      }
      case false => {
        data.map(dict.lookup(_).toDouble)
      }
    })
  }

  def dictSize: Int = dict.size
}

class S2SDataset(file: String, tokenizer: LSTMTokenizer,
                 val bosSymbol: String, val eosSymbol: String,
                 extEncDict: LSTMDict = null, extDecDict: LSTMDict = null)
  extends VarLenDataset {

  var encDict: LSTMDict = _
  var decDict: LSTMDict = _

  override protected def load(): Array[Data] = {
    val source = Source.fromFile(file)
    try {
      if (extEncDict == null) {
        encDict = new LSTMDict()
        encDict.update(bosSymbol)
        encDict.update(eosSymbol)
      }
      else encDict = extEncDict

      if (extDecDict == null) {
        decDict = new LSTMDict()
        decDict.update(bosSymbol)
        decDict.update(eosSymbol)
      }
      else decDict = extDecDict

      source.getLines().map(line => {
        val tokens = tokenizer.tokenize(line).toArray
        if (!(tokens(0).equals(bosSymbol) && tokens.last.equals(eosSymbol))) {
          throw new IllegalArgumentException("Invalid start/stop symbol")
        }
        // look for stop/start symbols
        val nextbos = tokens.drop(1).indexOf(bosSymbol) + 1
        if (nextbos == -1)
          throw new IllegalArgumentException()
        val firsteos = nextbos - 1
        if (!tokens(firsteos).equals(eosSymbol))
          throw new IllegalArgumentException()

        val seq1 = tokens.slice(0, nextbos)
        val seq2 = tokens.slice(nextbos, tokens.length)

        new S2SData(
          seq1.map(encDict.update(_).toDouble),
          seq2.map(decDict.update(_).toDouble)
        )
      }).toArray
    }

    finally {
      source.close()
    }
  }

  def encDictSize = encDict.size

  def decDictSize = decDict.size
}