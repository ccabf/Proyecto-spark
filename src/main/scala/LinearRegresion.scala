import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.regression.LinearRegression

import scala.io.Source

/**
 * Created by AZ on 31.01.2017
 */
object LinearRegresion {

  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","LinearRegresion", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()

    val training = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("resources/suic-ind.csv")
    println(training)

    val featureCols = Array("year","population","indexedCountry","indexedage")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(training)

    /**
     * Then we use the StringIndexer to take the column FNDX and make that the label.
     *  FNDX is the 1 or 0 indicator that shows whether the patient has cancer.
     * Like the VectorAssembler it will add another column to the dataframe.
     */


    val lr = new LinearRegression()
      .setMaxIter(200)
      .setRegParam(0.05)
      .setElasticNetParam(0.99)
      .setLabelCol("suicides_no")

    // Fit the model
    val lrModel = lr.fit(df2)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    lrModel.save("/Users/hubsantander/Desktop/Scala")


  }

}