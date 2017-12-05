package edu.uchicago.cs.encsel.query.tpch

import java.io.File

import edu.uchicago.cs.encsel.query.{HColumnPredicate, RowTempTable}
import edu.uchicago.cs.encsel.query.operator.HorizontalSelect

object HorizontalScan extends App {
  val schema = TPCHSchema.lineitemSchema
  //  val inputFolder = "/home/harper/TPCH/"
  val inputFolder = args(0)
  val colIndex = 5
  val suffix = ".parquet"
  val file = new File("%s%s%s".format(inputFolder, schema.getName, suffix)).toURI

  val recorder = new RowTempTable(schema);

  val thresholds = Array(6000, 8000, 17000, 36000, 50000, 53000, 63000, 69000)
  println(thresholds.map(scan(_)).mkString("\n"))

  def scan(threshold: Long): Long = {
    val predicate = new HColumnPredicate((value: Any) => value.asInstanceOf[Double] < threshold, colIndex)
    val start = System.currentTimeMillis()

    new HorizontalSelect().select(file, predicate, schema, Array(0, 1, 2, 3, 4))

    System.currentTimeMillis() - start
  }

}
