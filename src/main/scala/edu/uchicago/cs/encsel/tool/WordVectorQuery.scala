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

package edu.uchicago.cs.encsel.tool

import scala.io.Source

object WordVectorQuery extends App {

  val src = "/home/harper/Downloads/glove.42B.300d.txt"

  val word1 = "st."
  val word2 = "rd."
  val word3 = "road"
  val word4 = "street"
  val word5 = "avenue"

  Source.fromFile(src).getLines().foreach {
    line =>
      {
        val sp = line.split("\\s+")
        if (sp(0).equals(word1))
          println(line)
        if (sp(0).equals(word2))
          println(line)
        if (sp(0).equals(word3))
          println(line)
        if (sp(0).equals(word4))
          println(line)
        if (sp(0).equals(word5))
          println(line)
      }
  }
}