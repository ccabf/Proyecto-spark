import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession


object csv {
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
    val df = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("resources/Influenza-NY.csv").toDF()
    //println("Show")
    //df.show()
    //println("Schema")
    //df.printSchema()
    //df.show()
    val df2=df.drop("Row", "Week Ending Date", "Area", "Number_households", "Beds_adult_facility_care","Beds_hospital","County_Served_hospital", "Discharges_Other_Hospital_intervention", "Discharges_Respiratory_system_interventions",
      "Total_Charge_Other_Hospital_intervention","Total_Charge_Respiratory_system_interventions", "Unemp_rate", "Medianfamilyincome","Service_hospital")
    df2.limit(5).show()



    val featureCols = Array("Country","Year","Month","Season","Region","Week","Disease", "Avg household size","Population","Under_18","18-24","25-44","45-64","Above_65","Median_age")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Infected")


    val labelIndexer = new StringIndexer()
      .setInputCol("Country")
      .setOutputCol("indexedCountry")
      .fit(df2)

    val dfindex = labelIndexer.transform(df2)

    //dfindex.show()

    val labelIndexer2 = new StringIndexer()
      .setInputCol("Region")
      .setOutputCol("indexedRegion")
      .fit(dfindex)

    val dfindex2 = labelIndexer2.transform(dfindex)

    dfindex2.show()

    val labelIndexer3 = new StringIndexer()
      .setInputCol("Disease")
      .setOutputCol("indexedDisease")
      .fit(dfindex2)

    val dfindex3 = labelIndexer3.transform(dfindex2)


    val dfindex4=dfindex3.drop("Country","Season","Region","Disease")


    dfindex4.show()

    val model = new LogisticRegression().fit(dfindex4)
    val predictions = model.transform(dfindex4)


    dfindex.write.format("csv").option("header", "true").option("delimiter", ",").save("resources/salida-29-06-16.55")


  }

}