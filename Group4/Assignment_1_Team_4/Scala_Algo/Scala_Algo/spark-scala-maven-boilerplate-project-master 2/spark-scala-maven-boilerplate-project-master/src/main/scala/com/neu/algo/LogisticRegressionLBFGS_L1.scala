package com.neu.algo

import org.apache.spark._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater

/**
 * @author insignia
 */
object LogisticRegressionLBFGS_L1 {
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
        
        // Run training algorithm to build the model
        val svmAlg = new LogisticRegressionWithLBFGS().setNumClasses(10)
        svmAlg.optimizer.setNumIterations(100).setRegParam(0.1).setUpdater(new L1Updater)
        val modelL1 = svmAlg.run(training)
        
        
        // Compute raw scores on the test set.
        val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
          val prediction = modelL1.predict(features)
          (prediction, label)
        }
        
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val precision = metrics.precision
        println("Precision = " + precision)
  
  }
  
}