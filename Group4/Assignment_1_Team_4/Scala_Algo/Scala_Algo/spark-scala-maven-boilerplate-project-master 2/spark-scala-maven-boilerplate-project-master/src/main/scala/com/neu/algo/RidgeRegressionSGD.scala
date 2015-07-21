package com.neu.algo

import org.apache.spark._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors

/**
 * @author insignia
 */
object RidgeRegressionSGD {
  
  def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Spark Pi")
        /** Create the SparkContext */
        val sc = new SparkContext(conf)
        
        val Delimeter = ","
        val textFile = sc.textFile("/Users/insignia/Desktop/Big_Data_Analytics/Assignment/winequality-white.csv")
        val data = textFile.map { line =>
        val parts = line.split(Delimeter)
        LabeledPoint(parts(11).toDouble, Vectors.dense(parts.slice(1,10).map(x => x.toDouble).toArray))
        }
        // Split data into training (60%) and test (40%).
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        
        // Building the model
        val numIterations = 100
        val stepSize = 0.00001
        val regParam = .01
        val model = RidgeRegressionWithSGD.train(training, numIterations, stepSize, regParam)
        
        // Evaluate model on training examples and compute training error
        val valuesAndPreds = training.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
        println("training Mean Squared Error = " + MSE)
  }
  
}