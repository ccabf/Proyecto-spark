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
import org.apache.spark.sql.functions.col


object CsvCo {
  def main(args: Array[String]): Unit = {

    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)

    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "csv", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()


    val df = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("resources/who_suicide_statistics.csv").toDF()
    //println("Show")
    //df.show()
    //println("Schema")
    //df.printSchema()
    //df.show()
    //val df2=df.drop("Timestamp","Choose your gender","Age","What is your course?","Your current year of Study","What is your CGPA?","Marital status","Do you have Depression?","Do you have Anxiety?","Do you have Panic attack?","Did you seek any specialist for a treatment?")
    //df2.limit(5).show()

    val df_na= df.na.drop("Any")

    val labelIndexer = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("indexedCountry")
      .fit(df_na)

    val dfindex = labelIndexer.transform(df_na)

    //dfindex.show()

    val labelIndexer2 = new StringIndexer()
      .setInputCol("sex")
      .setOutputCol("indexedSex")
      .fit(dfindex)

    val dfindex2 = labelIndexer2.transform(dfindex)

    val labelIndexer3 = new StringIndexer()
      .setInputCol("age")
      .setOutputCol("indexedage")
      .fit(dfindex2)

    val dfindex3 = labelIndexer3.transform(dfindex)

    val dfindex4 = dfindex3.drop("age", "sex", "country")



    dfindex3.printSchema()

    dfindex4.write.format("csv").option("header", "true").option("delimiter", ",").save("resources/salida-01-07-10.38")

  }

}
