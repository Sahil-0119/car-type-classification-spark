
ThisBuild / scalaVersion := "2.13.18"

ThisBuild / version := "1.0"

ThisBuild / organization := "com.carclassification"

lazy val root = (project in file("."))
  .settings(
    name := "CarTypeClassification",

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "4.3.0",
      "org.apache.spark" %% "spark-mllib" % "4.3.0"
    )
  )
