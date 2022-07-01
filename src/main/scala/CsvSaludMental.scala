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


object CsvSaludMental {
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
      .option("inferSchema", "true").option("delimiter", ",").load("resources/Student-Mental-health.csv").toDF()
    //println("Show")
    //df.show()
    //println("Schema")
    //df.printSchema()
    //df.show()
    //val df2=df.drop("Timestamp","Choose your gender","Age","What is your course?","Your current year of Study","What is your CGPA?","Marital status","Do you have Depression?","Do you have Anxiety?","Do you have Panic attack?","Did you seek any specialist for a treatment?")
    //df2.limit(5).show()

    val labelIndexer = new StringIndexer()
      .setInputCol("Choose your gender")
      .setOutputCol("indexedGender")
      .fit(df)

    val dfindex = labelIndexer.transform(df)

    //dfindex.show()

    val labelIndexer2 = new StringIndexer()
      .setInputCol("What is your course?")
      .setOutputCol("indexedCourse")
      .fit(dfindex)

    val dfindex2 = labelIndexer2.transform(dfindex)

    //dfindex2.show()

    val labelIndexer3 = new StringIndexer()
      .setInputCol("Marital status")
      .setOutputCol("indexedMarital")
      .fit(dfindex2)

    val dfindex3 = labelIndexer3.transform(dfindex2)

    val labelIndexer4 = new StringIndexer()
      .setInputCol("Do you have Depression?")
      .setOutputCol("indexedDepr")
      .fit(dfindex3)

    val dfindex4 = labelIndexer4.transform(dfindex3)

    val labelIndexer5 = new StringIndexer()
      .setInputCol("Do you have Anxiety?")
      .setOutputCol("indexedAnx")
      .fit(dfindex4)

    val dfindex5 = labelIndexer5.transform(dfindex4)

    val labelIndexer6 = new StringIndexer()
      .setInputCol("Do you have Panic attack?")
      .setOutputCol("indexedPan")
      .fit(dfindex5)

    val dfindex6 = labelIndexer6.transform(dfindex5)

    val labelIndexer7 = new StringIndexer()
      .setInputCol("Did you seek any specialist for a treatment?")
      .setOutputCol("indexedSpec")
      .fit(dfindex6)

    val dfindex7 = labelIndexer7.transform(dfindex6)

    val labelIndexer8 = new StringIndexer()
      .setInputCol("Your current year of Study")
      .setOutputCol("indexedYear")
      .fit(dfindex7)

    val dfindex8 = labelIndexer8.transform(dfindex7)

    val labelIndexer9 = new StringIndexer()
      .setInputCol("What is your CGPA?")
      .setOutputCol("indexedCGPA")
      .fit(dfindex8)

    val dfindex9 = labelIndexer9.transform(dfindex8)

    val dfindex10 = dfindex9.na.drop("Any")


    val dfindex11 = dfindex10.drop("TimeStamp","What is your CGPA?","Choose your gender", "What is your course?", "Your current year of Study", "Marital status", "Do you have Depression?", "Do you have Anxiety?", "Do you have Panic attack?", "Did you seek any specialist for a treatment?")


    dfindex11.printSchema()

    dfindex11.write.format("csv").option("header", "true").option("delimiter", ",").save("resources/salida-30-06-16.58")



  }

}
