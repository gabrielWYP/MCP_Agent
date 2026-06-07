"""
Student model package — YOLOv8-Nano for knowledge distillation.

Public API:
    StudentModel: Full pipeline (backbone → neck → head → 7-key dict)
    CSPDarknetNano: Backbone building blocks
    PANet: FPN + PAN neck
    YOLOStudentHead: Decoupled detection head
    DecoupledHead: Per-level head with cls/reg stems
"""

from src.models.student.student_model import StudentModel
from src.models.student.backbone import (
    CSPDarknetNano,
    Conv_BN_SiLU,
    Bottleneck,
    C2f,
    SPPF,
)
from src.models.student.neck import PANet
from src.models.student.head import YOLOStudentHead, DecoupledHead

__all__ = [
    "StudentModel",
    "CSPDarknetNano",
    "Conv_BN_SiLU",
    "Bottleneck",
    "C2f",
    "SPPF",
    "PANet",
    "YOLOStudentHead",
    "DecoupledHead",
]
