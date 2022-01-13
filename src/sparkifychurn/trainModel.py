from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator



def train_lr_model(num_folds, train, features_to_scale):
    """ Set-up and fits cross-validated model pipeline.
    Defaults to penalized logistic regression.

    :param num_folds:
    :param train:
    :return:
    """

    si = StringIndexer(inputCol="gender", outputCol="gender_idx")

    ohe = OneHotEncoder(inputCol="gender_idx",
                        outputCol="gender_ohe")

    va_sc = VectorAssembler(inputCols=features_to_scale, outputCol="features_to_scale")

    sc = StandardScaler(inputCol="features_to_scale",
                        outputCol="scaled_features",
                        withMean=True,
                        withStd=True)

    va = VectorAssembler(inputCols=["gender_ohe", "scaled_features"],
                         outputCol="features")

    model = LogisticRegression(featuresCol="features", labelCol="churn", standardization=False)

    ml_pipeline = Pipeline(stages=[si, ohe, va_sc, sc, va, model])

    param_grid = ParamGridBuilder() \
        .addGrid(model.regParam, [0, 0.01, 0.05, 0.1, 0.015, 0.2]) \
        .addGrid(model.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]) \
        .build()

    model_crossval = CrossValidator(estimator=ml_pipeline,
                                    estimatorParamMaps=param_grid,
                                    evaluator=BinaryClassificationEvaluator(labelCol="churn",
                                                                            metricName="areaUnderPR"),
                                    numFolds=num_folds,
                                    seed=1234)

    cv_model = model_crossval.fit(train)

    return cv_model


def train_gbt_model(num_folds, train, features_to_scale):
    """ Set-up and fits cross-validated model pipeline for GBT classifier.

    :param num_folds:
    :param train:
    :return:
    """

    si = StringIndexer(inputCol="gender", outputCol="gender_idx")

    ohe = OneHotEncoder(inputCol="gender_idx",
                        outputCol="gender_ohe")

    va_sc = VectorAssembler(inputCols=features_to_scale, outputCol="features_to_scale")

    sc = StandardScaler(inputCol="features_to_scale",
                        outputCol="scaled_features",
                        withMean=True,
                        withStd=True)

    va = VectorAssembler(inputCols=["gender_ohe", "scaled_features"],
                         outputCol="features")

    model = GBTClassifier(featuresCol="features", labelCol="churn")

    ml_pipeline = Pipeline(stages=[si, ohe, va_sc, sc, va, model])

    param_grid = ParamGridBuilder() \
        .addGrid(model.maxDepth, [2, 5, 10, 15]) \
        .addGrid(model.maxIter, [10, 20, 40, 100, 200]) \
        .build()

    model_crossval = CrossValidator(estimator=ml_pipeline,
                                    estimatorParamMaps=param_grid,
                                    evaluator=BinaryClassificationEvaluator(labelCol="churn",
                                                                            metricName="areaUnderPR"),
                                    numFolds=num_folds,
                                    seed=1234)

    cv_model = model_crossval.fit(train)

    return cv_model






