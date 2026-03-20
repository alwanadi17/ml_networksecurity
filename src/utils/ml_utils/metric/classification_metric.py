from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.artifact_entity import ClassificationReportArtifact
from src.utils.utils import write_yaml_file

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd
import os
import sys

def classification_result(self, y_true: pd.Series, y_pred: pd.Series, report_file_path:str) -> ClassificationReportArtifact:
    try:
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        classification_report_dict = classification_report(y_true, y_pred, output_dict=True)

        report_to_save = {
            'overall_accuracy': accuracy,
            'overall_roc_auc': roc_auc,
            'overall_f1_score': f1,
            'overall_precision': precision,
            'overall_recall': recall,
            'macro_avg': classification_report_dict['macro avg'],
            'weighted_avg': classification_report_dict['weighted avg'],
            'class_specific': {
                'attack': classification_report_dict['-1'],
                'normal': classification_report_dict['1']
            }
        }

        logging.info(f"Saving classification report at {report_file_path}")
        write_yaml_file(report_file_path, report_to_save)

        classification_report_artifact = ClassificationReportArtifact(
            report_file_path=report_file_path,
            accuracy=accuracy,
            roc_auc=roc_auc,
            f1_score=f1,
            precision=precision,
            recall=recall
        )

        return classification_report_artifact
    except Exception as e:
        logging.error(f"Error occurred in classification_result: {e}")
        raise NetException(e, sys) from e