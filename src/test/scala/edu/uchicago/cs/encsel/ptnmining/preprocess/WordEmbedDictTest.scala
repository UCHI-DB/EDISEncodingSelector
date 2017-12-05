package edu.uchicago.cs.encsel.ptnmining.preprocess

import java.io.File
import java.nio.file.{Files, Paths}

import org.junit.Assert._
import org.junit.Test

object WordEmbedDictTest {
  val file = "/home/harper/Downloads/glove.840B.300d.txt"
}

class WordEmbedDictTest {

  @Test
  def testOutput: Unit = {
    if (Files.exists(Paths.get(new File(WordEmbedDictTest.file).toURI))) {
      val words = new WordEmbedDict(WordEmbedDictTest.file)
    }
  }

  @Test
  def testFind: Unit = {
    if (Files.exists(Paths.get(new File(WordEmbedDictTest.file).toURI))) {
      val words = new WordEmbedDict(WordEmbedDictTest.file)

      assertTrue(words.find("st.").isDefined)
    }
  }

  @Test
  def testCompare: Unit = {
    if (Files.exists(Paths.get(new File(WordEmbedDictTest.file).toURI))) {
      val words = new WordEmbedDict(WordEmbedDictTest.file)

      assertTrue(0.5 < words.compare("street", "avenue"))
    }
  }

  @Test
  def testAddPhrase: Unit = {
    if (Files.exists(Paths.get(new File(WordEmbedDictTest.file).toURI))) {
      val words = new WordEmbedDict(WordEmbedDictTest.file)

      assertTrue(words.find("good man").isEmpty)

      words.addPhrase("good man", Array("good", "man"))

      assertTrue(words.find("good man").isDefined)

      val result = words.find("good man").get
      val good = words.find("good").get
      val man = words.find("man").get

      assertEquals(result, good.add(man))
    }
  }
}