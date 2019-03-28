package skipgram

import scala.io.Source

object preprocessing{
  def article2Sequences(article:String) = {
    article.split("\\.\\s").map(sequence => sequence
      .replaceAll("<.*>","")
      .replaceAll("'.","")
      .map(_.toLower)
      .replaceAll("[^a-z]"," ")
      .split("\\s+"))
  }

  def getWordFrequency(words:Seq[String]) = {
    words.map(word => word -> words.count(_ == word)).toMap
  }
}
