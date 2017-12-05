package edu.uchicago.cs.ndnn

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp

class NodeTest {

  @Test
  def testForward: Unit = {

    val env = new NodeEnv {}
    val node1 = env.input("")
    val node2 = env.input("")
    val node3 = new DummyNode(node1, node2)
    val node4 = new DummyNode(node3)
    val node5 = new DummyNode(node3)
    val node6 = new DummyNode(node4)
    val node7 = new DummyNode(node2, node5, node6)

    node1.value = Nd4j.createUninitialized(Array(2, 3, 5)).assign(1)
    node2.value = Nd4j.createUninitialized(Array(4, 9, 7)).assign(2)

    env.forward

    assertEquals(node2.value, node7.value)
  }

  @Test
  def testBackward: Unit = {

    val env = new NodeEnv {}
    val node1 = env.input("node1")
    val node2 = env.input("node2")
    val node3 = new DummyNode(node1, node2)
    val node4 = new DummyNode(node3)
    val node5 = new DummyNode(node3)
    val node6 = new DummyNode(node4)
    val node7 = new DummyNode(node2, node5, node6)

    node1.set(Nd4j.zeros(3, 2, 7))
    node2.set(Nd4j.zeros(3, 2, 7))

    env.forward

    node7.grad = Nd4j.createUninitialized(Array(3, 2, 7)).assign(1)
    env.backward

    assertArrayEquals(Array(3, 2, 7), node1.grad.shape)
    assertArrayEquals(Array(3, 2, 7), node2.grad.shape)

    for (i <- 0 to 2; j <- 0 to 1; k <- 0 to 6) {
      assertEquals(2, node1.grad.getDouble(i, j, k), 0.001)
      assertEquals(3, node2.grad.getDouble(i, j, k), 0.001)
    }
  }
}

class InputTest {
  @Test
  def testCompute: Unit = {
    val env = new NodeEnv {}
    val input1 = env.input("")
    val input2 = new Node(input1) {
      def compute = this.value = input1.value
      def updateGrad = input1.grad.addi(this.grad)
    }

    val value = Nd4j.create(Array(1d, 2d, 3))

    input1.set(value)
    env.forward

    for (i <- 0 to 2) {
      assertEquals(input2.value.getDouble(i), value.getDouble(i), 0.001)
    }
  }

  @Test
  def testBackward: Unit = {
    val env = new NodeEnv {}
    val input1 = env.input("")
    val input2 = new Node(input1) {
      def compute = this.value = input1.value
      def updateGrad = input1.grad.addi(this.grad)
    }

    val grad = Nd4j.create(Array(1d, 2d, 3))

    input1.set(grad.dup())
    env.forward
    input2.grad = grad.dup()
    env.backward

    for (i <- 0 to 2) {
      assertEquals(grad.getDouble(i), input1.grad.getDouble(i), 0.001)
    }
  }
}

class AddTest {
  @Test
  def testCompute: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val input2 = env.input("")

    val add = new Add(input, input2)

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(1d, 0, 1))

    env.forward

    val expected = Nd4j.create(Array(Array(4d, 2, 6), Array(3d, 2, 2), Array(2d, 0, 1), Array(5d, 0, 3)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(add.value.getDouble(i, j), expected.getDouble(i, j), 0.0001)
    }

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))

    env.forward
    val expected2 = Nd4j.create(Array(Array(6d, 4, 10), Array(4d, 4, 2), Array(2d, 0, 0), Array(8d, 0, 4)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(expected2.getDouble(i, j), add.value.getDouble(i, j), 0.0001)
    }

    input.value = Nd4j.create(Array(3d, 2, 5, 2d, 2, 1, 1d, 0, 0, 4d, 0, 2), Array(2, 2, 3))
    input2.value = Nd4j.create(Array(4d, 0, 2))
    env.forward
    val expected3 = Nd4j.create(Array(7d, 2, 7, 6d, 2, 3, 5d, 0, 2, 8d, 0, 4), Array(2, 2, 3))
    for (i <- 0 to 1; j <- 0 to 1; k <- 0 to 2) {
      assertEquals("%d,%d,%d".format(i, j, k), expected3.getDouble(i, j, k), add.value.getDouble(i, j, k), 0.0001)
    }
  }

  @Test
  def testUpdateGrad: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val input2 = env.input("")

    val add = new Add(input, input2)

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(1d, 0, 1))

    env.forward
    add.grad = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    env.backward

    val grad1 = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    val grad2 = Nd4j.create(Array(13d, 5, 11))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(input.grad.getDouble(i, j), grad1.getDouble(i, j), 0.0001)
    }
    for (i <- 0 to 2) {
      assertEquals(input2.grad.getDouble(i), grad2.getDouble(i), 0.0001)
    }

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))

    env.forward
    add.grad = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    env.backward

    val grad3 = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(input.grad.getDouble(i, j), grad3.getDouble(i, j), 0.0001)
      assertEquals(input2.grad.getDouble(i, j), grad3.getDouble(i, j), 0.0001)
    }
  }
}

class MulTest {
  @Test
  def testCompute: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val input2 = env.input("")

    val mul = new Mul(input, input2)

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(1d, 0, 1))

    env.forward

    val expected = Nd4j.create(Array(Array(3d, 0, 5), Array(2d, 0, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(mul.value.getDouble(i, j), expected.getDouble(i, j), 0.0001)
    }

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))

    env.forward
    val expected2 = Nd4j.create(Array(Array(9d, 4, 25), Array(4d, 4, 1), Array(1d, 0, 0), Array(16d, 0, 4)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(expected2.getDouble(i, j), mul.value.getDouble(i, j), 0.0001)
    }

    input.value = Nd4j.create(Array(3d, 2, 5, 2d, 2, 1, 1d, 0, 0, 4d, 0, 2), Array(2, 2, 3))
    input2.value = Nd4j.create(Array(4d, 2, 2))
    env.forward
    val expected3 = Nd4j.create(Array(12d, 4, 10, 8d, 4, 2, 4, 0, 0, 16d, 0, 4), Array(2, 2, 3))
    for (i <- 0 to 1; j <- 0 to 1; k <- 0 to 2) {
      assertEquals("%d,%d,%d".format(i, j, k), expected3.getDouble(i, j, k), mul.value.getDouble(i, j, k), 0.0001)
    }
  }

  @Test
  def testUpdateGrad: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val input2 = env.input("")

    val mul = new Mul(input, input2)

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 2, 3), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(1d, 2, 1))

    env.forward
    mul.grad = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    env.backward

    val grad1 = Nd4j.create(Array(Array(1d, 2, 1), Array(2d, 4, 3), Array(4d, 4, 5), Array(6d, 0, 2)))
    val grad2 = Nd4j.create(Array(35d, 10, 27))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals("%d,%d".format(i, j), input.grad.getDouble(i, j), grad1.getDouble(i, j), 0.0001)
    }
    for (i <- 0 to 2) {
      assertEquals(input2.grad.getDouble(i), grad2.getDouble(i), 0.0001)
    }

    input.value = Nd4j.create(Array(Array(3d, 2, 5), Array(2d, 2, 1), Array(1d, 0, 0), Array(4d, 0, 2)))
    input2.value = Nd4j.create(Array(Array(5d, 4, 7), Array(4d, 4, 3), Array(3d, 2, 2), Array(6d, 2, 4)))

    env.forward
    mul.grad = Nd4j.create(Array(Array(1d, 1, 1), Array(2d, 2, 3), Array(4d, 2, 5), Array(6d, 0, 2)))
    env.backward

    val grad3 = Nd4j.create(Array(Array(5d, 4, 7), Array(8d, 8, 9), Array(12d, 4, 10), Array(36d, 0, 8)))
    val grad4 = Nd4j.create(Array(Array(3d, 2, 5), Array(4d, 4, 3), Array(4d, 0, 0), Array(24d, 0, 4)))
    for (i <- 0 to 3; j <- 0 to 2) {
      assertEquals(grad3.getDouble(i, j), input.grad.getDouble(i, j), 0.0001)
      assertEquals(grad4.getDouble(i, j), input2.grad.getDouble(i, j), 0.0001)
    }
  }
}

class SoftMaxTest {
  @Test
  def testCalculate: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val softmax = new SoftMax(input)

    input.set(Nd4j.create(Array(Array(1d, 2d, 3d, 4d, 5d), Array(2d, 7d, 6d, 2d, 3d), Array(1d, 1d, 2d, 2d, 3d))))
    val inputbackup = input.value.dup()

    env.forward

    val result = softmax.value
    assertArrayEquals(Array(3, 5), result.shape())
    val expected = Array(Array(0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865),
      Array(0.00481395, 0.71445362, 0.2628328, 0.00481395, 0.01308567),
      Array(0.06745081, 0.06745081, 0.1833503, 0.1833503, 0.49839779))
    for (i <- 0 until 3; j <- 0 until 5) {
      assertEquals(input.value.getDouble(i, j), inputbackup.getDouble(i, j), 0.01)
      assertEquals(expected(i)(j), result.getDouble(i, j), 0.01)
    }
  }

  @Test
  def testDerivative: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val softmax = new SoftMax(input)

    input.set(Nd4j.create(Array(Array(1d, 2d, 3d, 4d, 5d), Array(2d, 7d, 6d, 2d, 3d), Array(1d, 1d, 2d, 2d, 3d))))
    env.forward

    val grad = Nd4j.create(Array(Array(0, 0, 1d / 3, 0, 0), Array(0, 2d / 3, 0, 0, 0), Array(0, 0, 0, 0, 1.5 / 3)))

    val valbackup = softmax.value.dup()

    softmax.grad = grad
    env.backward
    val result = input.grad

    val expected = Array(Array(-0.00033464, -0.00090966, 0.02623681, -0.00672152, -0.01827098),
      Array(-0.0022929, 0.13600643, -0.1251879, -0.0022929, -0.00623274),
      Array(-0.01680867, -0.01680867, -0.04569069, -0.04569069, 0.12499872))
    for (i <- 0 until 3; j <- 0 until 5) {
      assertEquals(expected(i)(j), result.getDouble(i, j), 0.0001)
      assertEquals(valbackup.getDouble(i, j), softmax.value.getDouble(i, j), 0.0001)
    }

    assertFalse(softmax.value.isCleanedUp)
    assertFalse(softmax.grad.isCleanedUp)
  }
}

class SigmoidTest {

  @Test
  def testCompute: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val sigmoid = new Sigmoid(input)
    input.set(Nd4j.create(Array(Array(0.4, 0.9, -1.1, 3.8), Array(2.2, 1.7, 0.5, -0.7))))

    env.forward

    assertArrayEquals(Array(2, 4), sigmoid.value.shape())

    val expected = Nd4j.create(Array(Array(0.59868766, 0.7109495, 0.24973989, 0.97811873),
      Array(0.90024951, 0.84553473, 0.62245933, 0.33181223)))
    for (i <- 0 to 1; j <- 0 to 3) {
      assertEquals(expected.getDouble(i, j), sigmoid.value.getDouble(i, j), 0.001)
    }
  }

  @Test
  def testUpdateGrad: Unit = {
    val env = new NodeEnv {}
    val input = env.input("")
    val sigmoid = new Sigmoid(input)
    input.set(Nd4j.create(Array(Array(0.59868766, 0.7109495, 0.24973989, 0.97811873),
      Array(0.90024951, 0.84553473, 0.62245933, 0.33181223))))
    env.forward

    sigmoid.value = Nd4j.create(Array(Array(0.59868766, 0.7109495, 0.24973989, 0.97811873),
      Array(0.90024951, 0.84553473, 0.62245933, 0.33181223)))
    val grad = Nd4j.create(Array(Array(1, 0, 2, 4d), Array(2, 1, 7, 5d)))

    val valbackup = sigmoid.value.dup()
    sigmoid.grad = grad
    env.backward

    val expected = grad.muli(sigmoid.value.mul(sigmoid.value.sub(1).negi()))

    for (i <- 0 to 1; j <- 0 to 3) {
      assertEquals(expected.getDouble(i, j), input.grad.getDouble(i, j), 0.001)
      assertEquals(valbackup.getDouble(i, j), sigmoid.value.getDouble(i, j), 0.0001)
    }
  }
}

class ConcatTest {
  @Test
  def testCompute(): Unit = {
    val env = new NodeEnv {}
    val a = env.input("")
    val b = env.input("")

    a.set(Nd4j.create(Array(Array(1d, 0, 2), Array(2d, 2, 1), Array(3d, 4, 7))))
    b.set(Nd4j.create(Array(Array(3d, 9), Array(4d, 7), Array(5d, 1))))

    val concat = new Concat(a, b)
    env.forward

    assertArrayEquals(Array(3, 5), concat.value.shape)

    for (i <- 0 to 2) {
      for (j <- 0 to 2) {
        assertEquals(concat.value.getDouble(i, j), a.value.getDouble(i, j), 0.001)
      }
      for (j <- 3 to 4) {
        assertEquals(concat.value.getDouble(i, j), b.value.getDouble(i, j - 3), 0.001)
      }
    }
  }

  @Test
  def testUpdateGrad: Unit = {
    val env = new NodeEnv {}
    val a = env.input("")
    val b = env.input("")

    a.set(Nd4j.create(Array(Array(1d, 0, 2), Array(2d, 2, 1), Array(3d, 4, 7))))
    b.set(Nd4j.create(Array(Array(3d, 9), Array(4d, 7), Array(5d, 1))))

    val concat = new Concat(a, b)

    env.forward

    val grad = Nd4j.create(Array(Array(3d, 2, 4, 5, 1), Array(3d, 24, 9, 5, 6), Array(31d, 2, 41, 5, 6)))
    concat.grad = grad.dup()
    env.backward

    for (i <- 0 to 2) {
      for (j <- 0 to 2) {
        assertEquals(grad.getDouble(i, j), a.grad.getDouble(i, j), 0.001)
      }
      for (j <- 3 to 4) {
        assertEquals(grad.getDouble(i, j), b.grad.getDouble(i, j - 3), 0.001)
      }
    }
  }
}

class EmbedTest {

  @Test
  def testCompute: Unit = {
    val env = new NodeEnv {}
    val data = env.input("")
    val idx = env.input("")
    val embed = new Embed(idx, data)

    data.set(Nd4j.create(Array(
      Array(4d, 1, 2, 3),
      Array(5d, 6, 0, 7),
      Array(1d, 9, 3, 8),
      Array(3d, 2, 2, 1),
      Array(2d, 1, 2, 7))))
    idx.set(Array(2, 0, 1, 3, 4, 0, 1, 2))

    env.forward

    assertArrayEquals(Array(8, 4), embed.value.shape)

    val value = Nd4j.create(Array(
      Array(1d, 9, 3, 8), Array(4d, 1, 2, 3),
      Array(5d, 6, 0, 7),
      Array(3d, 2, 2, 1),
      Array(2d, 1, 2, 7),
      Array(4d, 1, 2, 3),
      Array(5d, 6, 0, 7),
      Array(1d, 9, 3, 8)))

    for (i <- 0 to 7; j <- 0 to 3) {
      assertEquals(value.getDouble(i, j), embed.value.getDouble(i, j), 0.001)
    }
  }

  @Test
  def testUpdateGrad: Unit = {
    val env = new NodeEnv {}
    val data = env.input("")
    val idx = env.input("")
    val embed = new Embed(idx, data)

    data.set(Nd4j.create(Array(
      Array(4d, 1, 2, 3),
      Array(5d, 6, 0, 7),
      Array(1d, 9, 3, 8),
      Array(3d, 2, 2, 1),
      Array(2d, 1, 2, 7))))
    idx.set(Array(2, 0, 1, 3, 4, 0, 1, 2))

    env.forward

    val grad = Nd4j.create(Array(
      Array(1d, 1, 2, 3),
      Array(2d, 2, 1, 7),
      Array(1d, 0, 2, 9),
      Array(3d, 0, 1, 1),
      Array(4d, 8, 5, 7),
      Array(4d, 9, 2, 3),
      Array(0d, 1, 0, 7),
      Array(1d, 9, 3, 5)))

    embed.grad = grad
    env.backward

    val datagrad = Nd4j.create(Array(
      Array(6d, 11, 3, 10),
      Array(1d, 1, 2, 16),
      Array(2d, 10, 5, 8),
      Array(3d, 0, 1, 1),
      Array(4d, 8, 5, 7)))

    assertArrayEquals(Array(5, 4), data.grad.shape)
    for (i <- 0 to 4; j <- 0 to 3) {
      assertEquals(datagrad.getDouble(i, j), data.grad.getDouble(i, j), 0.001)
    }
  }
}

class SliceTest {
  @Test
  def testCompute: Unit = {

  }

  @Test
  def testUpdateGrad: Unit = {

  }
}

class NewAxisTest {
  @Test
  def testCompute: Unit = {

  }

  @Test
  def testUpdateGrad: Unit = {

  }
}

class ReshapeTest {

}