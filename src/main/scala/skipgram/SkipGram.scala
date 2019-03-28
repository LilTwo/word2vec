package skipgram

import breeze.linalg.{DenseMatrix, DenseVector, sum, norm}

import scala.math.{exp, sqrt}

class SkipGram(embeddedDim: Int, trainingSet: Seq[(String, String)], lr: Double = 1e-2, batchSize:Int = 32) {
  val volcab = Set(trainingSet.map(_._1): _*)
  val idx2Word: Array[String] = volcab.toArray
  val word2Idx: Map[String, Int] = idx2Word.iterator.zipWithIndex.map { case (word, i) => word -> i }.toMap
  val Wi: DenseMatrix[Double] = DenseMatrix.fill(embeddedDim, volcab.size)((scala.util.Random.nextDouble() - 0.5) / volcab.size)
  val Wo: DenseMatrix[Double] = DenseMatrix.fill(volcab.size, embeddedDim)((scala.util.Random.nextDouble() - 0.5) / volcab.size)

  def fit(epochs: Int): Unit = {
    for (epoch <- 1 to epochs) {
      println(s"training... epoch:${epoch}")
      trainingSet.zipWithIndex.foreach { case ((center, context), i) => if (i % 10000 == 0) println(i); update(word2Idx(center), word2Idx(context)) }
    }
  }

  def update(centerIdx: Int, contextIdx: Int):DenseVector[Double] = {
    val ucenter = Wi(::, centerIdx)
    val samples = drawSample(5)
    val Z = sum(samples.map(idx => exp(Wo(idx, ::) * ucenter)))
    //ucenter 1*V * V*batch
    val dz = samples.map(idx => idx -> (exp(Wo(idx, ::) * ucenter) / Z - {
      if (idx == contextIdx) 1.0 else 0.0
    }))
    //    println(dz)

    //update Wo
    dz.foreach { case (idx, dzi) => Wo(idx, ::) -= ucenter.t * dzi * lr }
    //update Wi
    val WoSamples = DenseMatrix(samples.map(Wo(_, ::).t).toArray: _*)
    val ducenter = WoSamples.t * dz.map(_._2)
    ucenter -= lr * ducenter
  }

  def drawSample(n: Int):DenseVector[Int] = {
    DenseVector(0 until volcab.size: _*)
  }

  def predict(center: String): Seq[(String, Double)] = {
    val ucenter = Wi(::, word2Idx(center))
    val ucontexts = (0 until volcab.size).map(Wo(_, ::))
    val Z = ucontexts.map(ucontext => exp(ucontext * ucenter)).sum
    ucontexts.view.zip(idx2Word).map { case (ucontext, word) => word -> exp(ucontext * ucenter) / Z }.sortBy(-_._2).force
  }

  def getEmbbed(word: String) = Wi(::, word2Idx(word))


  def findMostSimiliar(word: String, n: Int = 5): Seq[(String, Double)] = {
    val ucenter = getEmbbed(word)
    (volcab - word).map(other => {
      val ucontext = getEmbbed(other); other -> ucenter.t * ucontext / norm(ucontext) / norm(ucenter)
    }).toArray.sortBy(-_._2).take(n)
  }

  def findRelation(a: String, b: String, c: String, n: Int = 5): Seq[(String, Double)] = {
    val xa = getEmbbed(a)
    val xb = getEmbbed(b)
    val xc = getEmbbed(c)
    (volcab - a - b - c).map(other => other -> (xb - xa + xc).t * getEmbbed(other) / norm(xb - xa + xc)).toArray.sortBy(_._2).take(n)
  }
}

object SkipGram extends App {
  def prepareSequence(sequence: Seq[String], windowSize: Int) = {
    val seqWithIdx = sequence.view.zipWithIndex
    seqWithIdx.flatMap { case (word, i) => seqWithIdx.slice(i - windowSize, i + windowSize + 1)
      .filter(_._2 != i)
      .map(word -> _._1)
    }.force
  }

  val words = Array("I", "like", "dogs", "are", "cats", "a", "dog", "are", "cat", "are", "cute", "too")
  val trainingSet = prepareSequence(words, 1)
  val sg = new SkipGram(10, trainingSet)
  println(sg.volcab)
  println()
  sg.fit(500)
  sg.predict("like").foreach(println)
  println(sg.findMostSimiliar("like", 3))
  println(sg.findRelation("dog", "cat", "dogs", 3))
}