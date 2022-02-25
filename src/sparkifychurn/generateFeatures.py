import pyspark.sql.functions as F


def generate_features(clean_df):
    """
    Perform feature engineering on existing log features.

    Args:
        clean_df: PySpark dataframe of cleaned log data

    Returns:
        user_summary: PySpark dataframe containing features aggregated to the user level.
    """

    clean_df = clean_df.withColumn("method_status", F.concat(F.col("method"), F.lit("_"), F.col("status")))

    session_summary = clean_df.groupby("userId", "sessionId") \
        .agg(((F.max("ts") - F.min("ts")) / 3600).alias("session_length_hours")) \
        .groupby("userId") \
        .agg(F.sum("session_length_hours").alias("sum_session_length_hours"),
             F.mean("session_length_hours").alias("avg_session_length_hours"))

    user_summary = clean_df.groupby("userId") \
        .agg(F.first("gender").alias('gender'),
             F.count("itemInSession").alias("count_items"),
             F.sum("length").alias("sum_length"),
             F.max(F.when(F.col("level") == "paid", 1).otherwise(0)).alias("paid"),
             F.countDistinct("sessionId").alias("session_count"),
             F.sum(F.when(F.col("page") == "Submit Downgrade", 1).otherwise(0)).alias("submit_downgrade_count"),
             F.sum(F.when(F.col("page") == "Thumbs Down", 1).otherwise(0)).alias("thumbs_down_count"),
             F.sum(F.when(F.col("page") == "Home", 1).otherwise(0)).alias("home_count"),
             F.sum(F.when(F.col("page") == "Downgrade", 1).otherwise(0)).alias("downgrade_count"),
             F.sum(F.when(F.col("page") == "Roll Advert", 1).otherwise(0)).alias("advert_count"),
             F.sum(F.when(F.col("page") == "Save Settings", 1).otherwise(0)).alias("save_settings_count"),
             F.max(F.when(F.col("page") == "Cancellation Confirmation", 1).otherwise(0)).alias("churn"),
             F.sum(F.when(F.col("page") == "About", 1).otherwise(0)).alias("about_count"),
             F.sum(F.when(F.col("page") == "Settings", 1).otherwise(0)).alias("settings_count"),
             F.sum(F.when(F.col("page") == "Add to Playlist", 1).otherwise(0)).alias("add_playlist_count"),
             F.sum(F.when(F.col("page") == "Add Friend", 1).otherwise(0)).alias("add_friend_count"),
             F.sum(F.when(F.col("page") == "NextSong", 1).otherwise(0)).alias("next_song_count"),
             F.sum(F.when(F.col("page") == "Thumbs Up", 1).otherwise(0)).alias("thumbs_up_count"),
             F.sum(F.when(F.col("page") == "Help", 1).otherwise(0)).alias("help_count"),
             F.sum(F.when(F.col("page") == "Upgrade", 1).otherwise(0)).alias("upgrade_count"),
             F.sum(F.when(F.col("page") == "Error", 1).otherwise(0)).alias("error_count"),
             F.max(F.when(F.col("page") == "Submit Upgrade", 1).otherwise(0)).alias("submit_upgrade"),
             F.sum(F.when((F.col("page") != "NextSong") & (F.col("page") != "Cancellation Confirmation"), 1).otherwise(0)).alias("non_song_interaction_count"),
             F.sum(F.when(F.col("method_status") == "PUT_200", 1).otherwise(0)).alias("PUT_200_count"),
             F.sum(F.when(F.col("method_status") == "GET_200", 1).otherwise(0)).alias("GET_200_count"),
             F.sum(F.when(F.col("method_status") == "PUT_307", 1).otherwise(0)).alias("PUT_307_count"),
             ((F.max("ts") - F.min("ts")) / (3600 * 24)).alias("tenure_days"))

    user_summary = user_summary.withColumn("thumbs_up_pct",
                                           F.when(F.col("thumbs_up_count") > 0,
                                                  F.col("thumbs_up_count") / (F.col("thumbs_up_count") + F.col(
                                                      "thumbs_down_count"))).otherwise(0)) \
        .withColumn("avg_items_session", F.col("count_items") / F.col("session_count")) \
        .withColumn("avg_songs_session", F.col("next_song_count") / F.col("session_count"))

    user_summary = user_summary.join(session_summary, "userId")

    user_summary = user_summary.withColumn("interaction_rate", F.col("count_items") / F.col("sum_session_length_hours")) \
        .withColumn("submit_downgrade_rate", F.col("submit_downgrade_count") / F.col("sum_session_length_hours")) \
        .withColumn("thumbs_down_rate", F.col("thumbs_down_count") / F.col("sum_session_length_hours")) \
        .withColumn("home_rate", F.col("home_count") / F.col("sum_session_length_hours")) \
        .withColumn("downgrade_rate", F.col("downgrade_count") / F.col("sum_session_length_hours")) \
        .withColumn("home_rate", F.col("home_count") / F.col("sum_session_length_hours")) \
        .withColumn("advert_rate", F.col("advert_count") / F.col("sum_session_length_hours")) \
        .withColumn("save_settings_rate", F.col("save_settings_count") / F.col("sum_session_length_hours")) \
        .withColumn("about_rate", F.col("about_count") / F.col("sum_session_length_hours")) \
        .withColumn("settings_rate", F.col("settings_count") / F.col("sum_session_length_hours")) \
        .withColumn("add_playlist_rate", F.col("add_playlist_count") / F.col("sum_session_length_hours")) \
        .withColumn("add_friend_rate", F.col("add_friend_count") / F.col("sum_session_length_hours")) \
        .withColumn("next_song_rate", F.col("next_song_count") / F.col("sum_session_length_hours")) \
        .withColumn("thumbs_up_rate", F.col("thumbs_up_count") / F.col("sum_session_length_hours")) \
        .withColumn("help_rate", F.col("help_count") / F.col("sum_session_length_hours")) \
        .withColumn("upgrade_rate", F.col("upgrade_count") / F.col("sum_session_length_hours")) \
        .withColumn("error_rate", F.col("error_count") / F.col("sum_session_length_hours")) \
        .withColumn("non_song_interaction_rate",
                    F.col("non_song_interaction_count") / F.col("sum_session_length_hours"))

    user_summary = user_summary.fillna(0)

    return user_summary
