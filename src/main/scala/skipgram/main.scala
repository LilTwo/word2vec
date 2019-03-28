package skipgram

import skipgram.preprocessing.{article2Sequences, getWordFrequency}

import scala.io.Source

object main extends App {
  val root = new java.io.File("./data")
  val files = root.list().take(300)
  val sequences = files.flatMap(file => article2Sequences(Source.fromFile(s"unsup/$file").mkString))
  val wordFreq = getWordFrequency(sequences.flatten[String])
  val rareWords = wordFreq.filter(_._2 <= 5).keys.toSet
  rareWords.foreach(println)
  val trainingSet = sequences.flatMap(sequence => SkipGram.prepareSequence(sequence.filterNot(rareWords.contains),1))
  val skipgram  = new SkipGram(16,trainingSet,0.5e-2)

  println(wordFreq("better"))
  println(wordFreq("good"))
  println(wordFreq("worse"))
  println(wordFreq("bad"))

  skipgram.fit(20)

  println(skipgram.findMostSimiliar("better",3))
  println(skipgram.findMostSimiliar("good",3))
  println(skipgram.findMostSimiliar("worse",3))
  println(skipgram.findMostSimiliar("bad",3))

}
