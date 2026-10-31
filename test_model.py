
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

os.environ["HADOOP_HOME"] = r"C:\Winutils"
os.environ["hadoop_home_dir"] = r"C:\Winutils"
os.environ["PATH"] = r"C:\Winutils\bin;" + os.environ["PATH"]

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("CarPredictionTest")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    model = PipelineModel.load("models/car-classification-pipeline")

    # Sample car input
    sample_car = [{
        "Year": 2022.0,
        "Engine HP": 250.0,
        "Engine Cylinders": 4.0,
        "highway MPG": 30.0,
        "city mpg": 22.0,
        "Engine Fuel Type": "premium unleaded (required)",
        "Transmission Type": "AUTOMATIC",
        "Driven_Wheels": "front wheel drive",
        "Vehicle Size": "Midsize"
    }]

    input_df = spark.createDataFrame(sample_car)

    # Generate prediction
    predictions = model.transform(input_df)

    predictions.select(
        "prediction",
        "probability"
    ).show(truncate=False)

    # Convert predicted index into vehicle style
    label_stage = model.stages[0]
    predicted_index = int(
        predictions.select("prediction").first()["prediction"]
    )

    predicted_style = label_stage.labels[predicted_index]

    print("\n================================")
    print("PREDICTED VEHICLE STYLE:", predicted_style)
    print("================================")

except Exception as e:
    print("Prediction failed:", str(e))

finally:
    spark.stop()
