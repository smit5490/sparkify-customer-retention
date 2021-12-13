import pyspark.sql.functions as F

def clean_logs(df):
    # Convert unixtimestamp to human readable
    df = df.withColumn("ts", F.to_timestamp(F.from_unixtime(F.substring(F.col("ts"), 0, 10))).cast("long"))

    # Remove logs without users:
    df = df.filter(F.trim(F.col("userId")) != "")

    return df
