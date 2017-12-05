package edu.uchicago.cs.ndnn

/**
 * Created by Cathy on 2/18/2017.
 */
class DummyNode(ins: Node*) extends Node(ins: _*) {
  val first = ins(0)
  def compute = this.value = first.value
  def updateGrad = ins.foreach(i => i.grad.addi(this.grad))
}
