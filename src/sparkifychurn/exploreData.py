import pandas as pd
import pyspark.sql.functions as F



def summarize_missing_vals(df):
    """ Analyzes features for null, nan, and empty string values.

    Args:
        df: PySpark dataframe

    Returns:
        missing_summary: Pandas dataframe containing missing value information for each column.

    """

    missing_cnt_list = []
    missing_pct_list = []
    empty_cnt_list = []
    empty_pct_list = []
    total_cnt = df.count()

    for feature in df.columns:
        if df.select(feature).dtypes[0][1] == "timestamp":
            miss_cnt = df.filter(F.isnull(feature)).count()
            missing_cnt_list.append(miss_cnt)
            missing_pct_list.append(round((miss_cnt / total_cnt) * 100, 2))

        else:
            miss_cnt = df.filter(F.isnull(feature) | F.isnan(feature)).count()
            missing_cnt_list.append(miss_cnt)
            missing_pct_list.append(round((miss_cnt / total_cnt) * 100, 2))

        if df.select(feature).dtypes[0][1] == "string":
            empty_cnt = df.filter(F.trim(F.col(feature)) == "").count()
            empty_cnt_list.append(empty_cnt)
            empty_pct_list.append(round((empty_cnt / total_cnt) * 100, 2))
        else:
            empty_cnt_list.append(0)
            empty_pct_list.append(0)
        print("****Feature {} complete.****".format(feature))

    missing_summary = pd.DataFrame({"feature": df.columns,
                                    "missing_count": missing_cnt_list,
                                    "missing_percentage": missing_pct_list,
                                    "empty_string_count": empty_cnt_list,
                                    "empty_percentage": empty_pct_list})
    return missing_summary