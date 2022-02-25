import pyspark.sql.functions as F

def clean_logs(df):
    # Convert timestamp
    df = df.withColumn("ts", F.to_timestamp(F.from_unixtime(F.substring(F.col("ts"), 0, 10))).cast("long"))

    # Remove logs without users:
    df = df.filter(F.trim(F.col("userId")) != "").filter(F.col("userId").isNotNull())

    # Set all null interaction lengths to zero
    df = df.fillna(value=0, subset=["length"])

    return df
