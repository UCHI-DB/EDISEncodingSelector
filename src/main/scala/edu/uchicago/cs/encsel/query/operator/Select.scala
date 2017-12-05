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

import java.net.URI

import edu.uchicago.cs.encsel.dataset.parquet.ParquetReaderHelper
import edu.uchicago.cs.encsel.dataset.parquet.ParquetReaderHelper.ReaderProcessor
import edu.uchicago.cs.encsel.query.{RowTempTable, _}
import edu.uchicago.cs.encsel.query.util.DataUtils
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.column.impl.ColumnReaderImpl
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.hadoop.Footer
import org.apache.parquet.hadoop.metadata.BlockMetaData
import org.apache.parquet.schema.MessageType

import scala.collection.JavaConversions._

trait Select {

  def select(input: URI, p: Predicate, schema: MessageType,
             projectIndices: Array[Int]): TempTable
}


class VerticalSelect extends Select {
  override def select(input: URI, p: Predicate, schema: MessageType,
                      projectIndices: Array[Int]): TempTable = {

    val vp = p.asInstanceOf[VPredicate]
    val recorder = new ColumnTempTable(schema)

    ParquetReaderHelper.read(input, new ReaderProcessor {
      override def processFooter(footer: Footer): Unit = {}

      override def processRowGroup(version: ParsedVersion, meta: BlockMetaData, rowGroup: PageReadStore): Unit = {
        val columns = schema.getColumns.zipWithIndex.map(col => new ColumnReaderImpl(col._1,
          rowGroup.getPageReader(col._1), recorder.getConverter(col._2).asPrimitiveConverter(), version))

        vp.leaves.foreach(leaf => {
          leaf.setColumn(columns(leaf.colIndex))
        })

        val bitmap = vp.bitmap

        projectIndices.map(columns(_)).foreach(col => {
          for (count <- 0L until rowGroup.getRowCount) {
            if (bitmap.test(count)) {
              col.writeCurrentValueToConverter()
            } else {
              col.skip()
            }
            col.consume()
          }
        })
      }
    })

    return recorder
  }
}

class HorizontalSelect extends Select {
  override def select(input: URI, p: Predicate, schema: MessageType,
                      projectIndices: Array[Int]): TempTable = {

    val hp = p.asInstanceOf[HPredicate]
    val recorder = new RowTempTable(schema)
    ParquetReaderHelper.read(input, new ReaderProcessor {
      override def processFooter(footer: Footer): Unit = {}

      override def processRowGroup(version: ParsedVersion, meta: BlockMetaData, rowGroup: PageReadStore): Unit = {
        val columns = schema.getColumns.zipWithIndex.map(col => new ColumnReaderImpl(col._1,
          rowGroup.getPageReader(col._1), recorder.getConverter(col._2).asPrimitiveConverter(), version))

        hp.leaves.foreach(leaf => {
          leaf.setColumn(columns(leaf.colIndex))
        })

        for (count <- 0L until rowGroup.getRowCount) {
          recorder.start()
          projectIndices.map(columns(_)).foreach(col => {
            if (hp.value) {
              col.writeCurrentValueToConverter()
            } else {
              col.skip()
            }
            col.consume()
          })
          recorder.end()
        }
      }
    })
    return recorder
  }
}
