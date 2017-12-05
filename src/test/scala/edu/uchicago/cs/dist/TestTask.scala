package edu.uchicago.cs.dist

import scala.collection.mutable

/**
  * Created by harper on 5/4/17.
  */
class TestTask extends DefaultTask {

  val done = new mutable.HashSet[Int]()

  override def spawn(numPiece: Int): Seq[TaskPiece] = {
    (0 until 10).map(i => new TestTaskPiece(this, i))
  }

  override def collect(piece: TaskPiece): Unit = {
    if (piece.isInstanceOf[TestTaskPiece]) {
      val tpiece = piece.asInstanceOf[TestTaskPiece]
      done += tpiece.number
    }
  }
}

class TestTaskPiece(task: Task, val number: Int) extends DefaultTaskPiece(task) {

  override def execute(): Unit = {
    this.result = this.id
  }
}
