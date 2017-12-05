package edu.uchicago.cs.encsel.query.tpch

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.parquet.VersionParser
import org.apache.parquet.column.page.{DataPage, PageReadStore}
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HiddenFileFilter

import scala.collection.JavaConversions._

object ReadTPCH extends App {
  val folder = args(0)
  val schema = TPCHSchema.customerSchema
  val file = "%s%s.parquet".format(folder,schema.getName)
  val conf = new Configuration
  val path = new Path(file)
  val fs = path.getFileSystem(conf)
  val statuses = fs.listStatus(path, HiddenFileFilter.INSTANCE).toIterator.toList
  val footers = ParquetFileReader.readAllFootersInParallelUsingSummaryFiles(conf, statuses, false)
  if (footers.isEmpty)
    throw new IllegalArgumentException();

  for (footer <- footers) {
    val version = VersionParser.parse(footer.getParquetMetadata.getFileMetaData.getCreatedBy)
    val fileReader = ParquetFileReader.open(conf, footer.getFile, footer.getParquetMetadata)
    var blockCounter = 0
    val cols = footer.getParquetMetadata.getFileMetaData.getSchema.getColumns

    var rowGroup: PageReadStore = null;
    var dataPage: DataPage = null

    rowGroup = fileReader.readNextRowGroup()
    while (rowGroup != null) {
      println("Row Group:" + blockCounter)
      val blockMeta = footer.getParquetMetadata.getBlocks.get(blockCounter)
      // Read each column
      for (cd <- cols) {
        println("Column:" + ","+cd.getPath.mkString(","))
        val pageReader = rowGroup.getPageReader(cd)
        var pageCounter = 0
        dataPage = pageReader.readPage()
        while (dataPage != null) {
          println("Page %d, count %d".format(pageCounter, dataPage.getValueCount))
          pageCounter += 1
          dataPage = pageReader.readPage()
        }
      }
      blockCounter += 1
      rowGroup = fileReader.readNextRowGroup()
    }
  }
}
