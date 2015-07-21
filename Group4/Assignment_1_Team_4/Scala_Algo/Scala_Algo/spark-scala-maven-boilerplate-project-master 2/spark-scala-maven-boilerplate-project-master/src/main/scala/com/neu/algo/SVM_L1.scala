package com.neu.algo

import org.apache.spark._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater

/**
 * @author insignia
 */
object SVM_L1 {
  
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi")
        /** Create the SparkContext */
        val sc = new SparkContext(conf)
        
        val Delimeter = ","
        val textFile = sc.textFile("/Users/insignia/Desktop/Big_Data_Analytics/Assignment/winequality-white.csv")
        val data = textFile.map { line =>
        val parts = line.split(Delimeter)
        LabeledPoint(parts(11).toDouble, Vectors.dense(parts.slice(0,10).map(_.toDouble)))
        }
        
        
        // Split data into training (60%) and test (40%).
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        // Run training algorithm to build the model
        val numIterations = 1000
        val stepSize  = .001
        val regParam = .01
        val svmAlg = new SVMWithSGD()

        svmAlg.optimizer.setNumIterations(100).setRegParam(1.0).setUpdater(new L1Updater).setStepSize(.001).setRegParam(.01)
        
        val modelL1 = svmAlg.run(training)
         
        // Clear the default threshold. 
        modelL1.clearThreshold() 
         
        // Compute raw scores on the test set. 
        val scoreAndLabels = test.map { point =>
        val score = modelL1.predict(point.features) 
        (score, point.label) } 
         
        // Get evaluation metrics. 
        val metrics = new BinaryClassificationMetrics(scoreAndLabels) 
        val auROC = metrics.areaUnderROC() 
         println("Area under ROC = " + auROC) 
        }
  
}