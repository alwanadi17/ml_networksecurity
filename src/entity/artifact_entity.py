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
class BaseDataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str
    
@dataclass
class GenericDataTransformationArtifact(BaseDataTransformationArtifact):
    pass

@dataclass
class CatBoostDataTransformationArtifact(BaseDataTransformationArtifact):
    data_config_file_path: str

# Main transformation artifact 
# that freely holds either generic or catboost transformation artifacts, depending on the use case
@dataclass
class DataTransformationArtifact:
    generic_data_transformation_artifact: Optional[GenericDataTransformationArtifact] = None
    catboost_data_transformation_artifact: Optional[CatBoostDataTransformationArtifact] = None

@dataclass
class ClassificationReportArtifact:
    accuracy: float
    roc_auc: float
    f1_score: float
    precision: float
    recall: float

@dataclass
class ModelTrainerArtifact:
    model_name: str
    params: Dict[str, Any]

    model_file_path: str
    model_params_file_path: str
    train_report_file_path: str
    test_report_file_path: str
    
    train_classification_report_artifact: ClassificationReportArtifact
    test_classification_report_artifact: ClassificationReportArtifact
