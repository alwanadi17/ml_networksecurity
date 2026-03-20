from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataIngestionArtifact:
    raw_file_path: str
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str
    
@dataclass
class GenericDataTransformationArtifact(DataTransformationArtifact):
    pass

@dataclass
class CatBoostDataTransformationArtifact(DataTransformationArtifact):
    data_config_file_path: str

@dataclass
class ClassificationReportArtifact:
    report_file_path: str
    accuracy: float
    roc_auc: float
    f1_score: float
    precision: float
    recall: float

@dataclass
class ModelTrainerArtifact:
    model_file_path: str
    model_config_file_path: str
    classification_report_train_artifact: ClassificationReportArtifact
    classification_report_test_artifact: ClassificationReportArtifact
    