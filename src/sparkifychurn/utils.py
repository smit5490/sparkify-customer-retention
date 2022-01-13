from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *


first_element = F.udf(lambda v: float(v[1]), FloatType())


def union_all(*dfs):
    return reduce(DataFrame.unionAll, dfs)


def train_test_stratified_split(df, feat, weights, seed):
    outcome_vals = df.select(feat).drop_duplicates().toPandas()[feat].tolist()

    train_ls = []
    test_ls = []
    for outcome in outcome_vals:
        tmp = df.filter(F.col("churn") == outcome).cache()
        train_tmp, test_tmp = tmp.randomSplit(weights, seed=seed)
        train_ls.append(train_tmp)
        test_ls.append(test_tmp)

    train = union_all(*train_ls)
    test = union_all(*test_ls)

    return train, test


