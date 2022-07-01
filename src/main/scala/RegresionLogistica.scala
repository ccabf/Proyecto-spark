import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Level
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{col, not}
import org.apache.spark.sql.functions.col

object RegresionLogistica {
  def main(args: Array[String]): Unit = {

    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)

    //Creando el contexto del Servidor
    val sc = new SparkContext("local","csv", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()


    val df_log = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("resources/MentalIndex.csv").toDF()
    //println("Show")
    //df.show()
    //println("Schema")
    //df.printSchema()
    //df.show()

/*
    dfindex4.withColumn("Year", col("Year").cast("Double"))
    dfindex4.withColumn("Month", col("Month").cast("Double"))
    dfindex4.withColumn("Week", col("Week").cast("Double"))
    dfindex4.withColumn("Avg household size", col("Avg household size").cast("Double"))
    dfindex4.withColumn("Population", col("Population").cast("Double"))
    dfindex4.withColumn("Under_18", col("Under_18").cast("Double"))
    dfindex4.withColumn("18-24", col("18-24").cast("Double"))
    dfindex4.withColumn("25-44", col("25-44").cast("Double"))
    dfindex4.withColumn("45-64", col("45-64").cast("Double"))
    dfindex4.withColumn("Above_65", col("Above_65").cast("Double"))
    dfindex4.withColumn("Median_age", col("Median_age").cast("Double"))
    dfindex4.withColumn("indexedCountry", col("indexedCountry").cast("Double"))
    dfindex4.withColumn("indexedRegion", col("indexedRegion").cast("Double"))
    dfindex4.withColumn("indexedDisease", col("indexedDisease").cast("Double"))
    dfindex4.withColumn("Infected", col("Infected").cast("Double"))
*/


    val featureCols = Array("Age","indexedGender","indexedCourse","indexedMarital","indexedAnx","indexedPan","indexedSpec","indexedYear","indexedCGPA")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df_log)



    /**
     * Then we use the StringIndexer to take the column FNDX and make that the label.
     *  FNDX is the 1 or 0 indicator that shows whether the patient has cancer.
     * Like the VectorAssembler it will add another column to the dataframe.
     */
    val labelIndexer = new StringIndexer().setInputCol("indexedDepr").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)

    val model = new LogisticRegression().fit(df3)
    val predictions = model.transform(df3)

    /**
     *  Now we print it out.  Notice that the LR algorithm added a “prediction” column
     *  to our dataframe.   The prediction in almost all cases will be the same as the label.  That is
     * to be expected it there is a strong correlation between these values.  In other words
     * if the chance of getting cancer was not closely related to these variables then LR
     * was the wrong model to use.  The way to check that is to check the accuracy of the model.
     *  You could use the BinaryClassificationEvaluator Spark ML function to do that.
     * Adding that would be a good exercise for you, the reader.
     */
    //predictions.select ("features", "label", "prediction").show(200)


    val lp = predictions.select( "label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter(col ("label") === col ("prediction")).count()
    val wrong = lp.filter(not(col ("label") === col ("prediction"))).count()
    val truep = lp.filter(col("prediction") === 0.0).filter(col ("label") === col ("prediction")).count()
    val falseN = lp.filter(col("prediction") === 0.0).filter(not(col ("label") === col ("prediction"))).count()
    val falseP = lp.filter(col("prediction") === 1.0).filter(not(col ("label") === col ("prediction"))).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble

    println("ratioWrong: "+ ratioWrong)
    println("ratioCorrect: "+ ratioCorrect)


  }

}