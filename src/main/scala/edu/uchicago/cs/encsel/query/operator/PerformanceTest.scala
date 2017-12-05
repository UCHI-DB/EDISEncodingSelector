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
 * under the License.
 *
 * Contributors:
 *     Hao Jiang - initial API and implementation
 */

package edu.uchicago.cs.encsel.query.operator

import scala.util.Random

object PerformanceTest extends App {

  val size = 100000000

  val data = new Array[Int](size)
  // Prepare data
  for (i <- 0 until size)
    data(i) = Random.nextInt()

  // Generate access index
  val accessSize = 10000000
  val index = new Array[Int](accessSize)

  val fetch = new Array[Int](accessSize)


  for (n <- 0 to 10) {
    for (i <- 0 until accessSize)
      index(i) = Random.nextInt(size)
    val start = System.currentTimeMillis()
    randomAccess(index, fetch)
    println(System.currentTimeMillis() - start)

    index.sortBy(i => i)

    val start2 = System.currentTimeMillis()
    sequentialAccess(index, fetch)
    println(System.currentTimeMillis() - start2)
  }

  def randomAccess(index: Array[Int], fetch: Array[Int]) = {
    for (i <- 0 until index.size)
      fetch(i) = data(index(i))
  }

  def sequentialAccess(index: Array[Int], fetch: Array[Int]) = {
    for (i <- 0 until index.size)
      fetch(i) = data(index(i))
  }

}
