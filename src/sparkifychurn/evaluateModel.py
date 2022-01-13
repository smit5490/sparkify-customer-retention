from sparkifychurn import utils
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, confusion_matrix, \
    classification_report
import plotly.express as px


def evaluate_model(model, model_name, output_path, train, test):
    """
    :param model:
    :param model_name:
    :param output_path:
    :param train:
    :param test:
    :return:
    """

    train_results = model.transform(train)
    test_results = model.transform(test)

    train_preds = train_results.select("churn", "prediction",
                                       utils.first_element("probability").alias("prob")).toPandas()
    test_preds = test_results.select("churn", "prediction", utils.first_element("probability").alias("prob")).toPandas()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(font_scale=1.25)
    sns.heatmap(confusion_matrix(train_preds["prediction"], train_preds["churn"]),
                cmap="Blues", annot=True, cbar=False, fmt='d', ax=ax)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title("Training Set Confusion Matrix")
    fig.savefig("{}/{}_train_cm.png".format(output_path, model_name))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(font_scale=1.25)
    sns.heatmap(confusion_matrix(test_preds["prediction"], test_preds["churn"]),
                cmap="Blues", annot=True, cbar=False, fmt='d', ax=ax)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title("Test Set Confusion Matrix")
    fig.savefig("{}/{}_test_cm.png".format(output_path, model_name))

    train_fpr, train_tpr, train_thresholds = roc_curve(train_preds["churn"], train_preds["prob"], pos_label=1)
    test_fpr, test_tpr, test_thresholds = roc_curve(test_preds["churn"], test_preds["prob"], pos_label=1)

    train_roc_auc = round(auc(train_fpr, train_tpr), 3)
    test_roc_auc = round(auc(test_fpr, test_tpr), 3)
    print("{} - Train Set AUROC: {}".format(model_name, train_roc_auc))
    print("{} - Test Set AUROC: {}".format(model_name, test_roc_auc))
    print("\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        train_fpr,
        train_tpr,
        color="blue",
        label="Train ROC (area = %0.2f)" % train_roc_auc,
    )
    ax.plot(
        test_fpr,
        test_tpr,
        color="red",
        label="Test ROC (area = %0.2f)" % test_roc_auc,
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic for {}".format(model_name))
    ax.legend(loc="lower right")
    fig.savefig("{}/{}_ROC.png".format(output_path, model_name))

#    train_f1 = round(f1_score(train_preds["churn"], train_preds["prediction"]), 3)
#    test_f1 = round(f1_score(test_preds["churn"], test_preds["prediction"]), 3)
#    print("{} - Train F1-Score: {}".format(model_name, train_f1))
#    print("{} - Test F1-Score: {}".format(model_name, test_f1))

    train_precision, train_recall, train_thresholds = precision_recall_curve(train_preds["churn"], train_preds["prob"])
    test_precision, test_recall, test_thresholds = precision_recall_curve(test_preds["churn"], test_preds["prob"])
    train_avg_precision = round(average_precision_score(train_preds["churn"], train_preds["prob"]), 3)
    test_avg_precision = round(average_precision_score(test_preds["churn"], test_preds["prob"]), 3)
    print("{} - Train Average Precision: {}".format(model_name, train_avg_precision))
    print("{} - Test Average Precision: {}".format(model_name, test_avg_precision))
    print("\n")
    train_no_skill_precision = round(train_preds["churn"].sum() / train_preds.shape[0], 3)
    test_no_skill_precision = round(test_preds["churn"].sum() / test_preds.shape[0], 3)
    print("{} - Train No Skill Precision: {}".format(model_name, train_no_skill_precision))
    print("{} - Test No Skill Precision: {}".format(model_name, test_no_skill_precision))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        train_recall,
        train_precision,
        color="blue",
        label="Train (Avg Precision = %0.2f)" % round(train_avg_precision, 2),
    )
    ax.plot(
        test_recall,
        test_precision,
        color="red",
        label="Test (Avg Precision = %0.2f)" % round(test_avg_precision, 2),
    )
    ax.plot([0, 1], [test_no_skill_precision, test_no_skill_precision], color="black", linestyle="--",
            label="No Skill Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curve for Logistic Regression")
    ax.legend(loc="lower right")
    fig.savefig("{}/{}_PR.png".format(output_path, model_name))

    print("\n ****Classification Report on Train Data***")
    print(classification_report(train_preds["churn"], train_preds["prediction"]))

    print("\n ****Classification Report on Test Data***")
    print(classification_report(test_preds["churn"], test_preds["prediction"]))

    px.line(x=train_recall[:-1],
            y=train_precision[:-1],
            hover_name=train_thresholds,
            title="Interactive Precision-Recall Curve",
            labels={"x": "Recall", "y": "Precision"},
            width=750)
    plt.show();
