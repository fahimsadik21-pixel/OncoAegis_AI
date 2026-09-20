"""
Registry of internal specialist imaging models.

The registry describes model capabilities; it does not load model weights.
Availability is checked separately so a missing checkpoint cannot silently
turn into a routing decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_modality(value: object) -> str:
    """Normalize user/API modality aliases to the internal vocabulary."""

    raw_value = getattr(value, "value", value)

    normalized = " ".join(
        str(raw_value or "").strip().upper().replace("-", " ").split()
    )

    aliases = {
        "US": "ULTRASOUND",
        "U S": "ULTRASOUND",
        "U/S": "ULTRASOUND",
        "SONOGRAPHY": "ULTRASOUND",
        "MR": "MRI",
    }

    return aliases.get(normalized, normalized)


def normalize_organ(value: object) -> str:
    return " ".join(
        str(value or "").strip().lower().replace("_", " ").split()
    )


def normalize_task(value: object) -> str:
    raw_value = getattr(value, "value", value)

    normalized = "_".join(
        str(raw_value or "").strip().lower().replace("-", " ").split()
    )

    aliases = {
        "detection": "lesion_detection",
        "nodule_detection": "lesion_detection",
        "localization": "lesion_detection",
        "segment": "segmentation",
        "classify": "classification",
    }

    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class ModelSpec:

    model_id: str
    name: str
    modality: str
    organ: str
    tasks: tuple[str, ...]
    input_type: str
    outputs: tuple[str, ...]
    status: str = "available"
    checkpoint_path: str | None = None
    model_kind: str | None = None
    dataset: str | None = None
    evidence_level: str = "research"
    clinical_validation: bool = False
    priority: int = 100


    def __post_init__(self):

        object.__setattr__(
            self,
            "modality",
            normalize_modality(self.modality),
        )

        object.__setattr__(
            self,
            "organ",
            normalize_organ(self.organ),
        )

        object.__setattr__(
            self,
            "tasks",
            tuple(
                normalize_task(task)
                for task in self.tasks
            ),
        )


    @property
    def checkpoint_available(self):

        if not self.checkpoint_path:
            return True

        return (
            _PROJECT_ROOT / self.checkpoint_path
        ).is_file()


    def to_dict(self):

        rendered = asdict(self)

        rendered["task"] = list(self.tasks)
        rendered["tasks"] = list(self.tasks)

        rendered["input"] = self.input_type

        rendered["output"] = list(self.outputs)
        rendered["outputs"] = list(self.outputs)

        rendered["availability"] = (
            "available"
            if self.checkpoint_available
            else "checkpoint_missing"
        )

        return rendered



_MODEL_SPECS: tuple[ModelSpec, ...] = (


    # =========================
    # LUNA16 CT
    # =========================

    ModelSpec(
        model_id="luna16_lung_segmentation",

        name="LUNA16 Lung Segmentation Model",

        modality="CT",

        organ="lung",

        tasks=("segmentation",),

        input_type="processed 2D CT slice",

        outputs=(
            "lung mask",
        ),

        checkpoint_path=
            "checkpoints/luna16/luna16_unet_best.pt",

        model_kind="2D U-Net",

        dataset="LUNA16",

        priority=10,
    ),



    ModelSpec(
        model_id="luna16_nodule_detector",

        name="LUNA16 Annotated-Nodule Patch Classifier",

        modality="CT",

        organ="lung",

        tasks=("lesion_detection",),

        input_type=
            "nodule-centred 2D CT patch",

        outputs=(
            "annotated-nodule patch likelihood",
        ),

        checkpoint_path=
            "checkpoints/luna16_nodule/"
            "luna16_nodule_classifier_best.pt",

        model_kind=
            "2D patch classifier",

        dataset="LUNA16",

        priority=20,
    ),



    # =========================
    # BUSI Ultrasound
    # =========================


    ModelSpec(
        model_id="busi_breast_segmentation",

        name="BUSI Breast Lesion Segmentation",

        modality="ULTRASOUND",

        organ="breast",

        tasks=("segmentation",),

        input_type=
            "grayscale ultrasound image",

        outputs=(
            "lesion mask",
        ),

        checkpoint_path=
            "checkpoints/busi/"
            "busi_multitask_unet_best.pt",

        model_kind=
            "multi-task 2D U-Net",

        dataset="BUSI",

        priority=10,
    ),



    ModelSpec(
        model_id="busi_breast_classifier",

        name="BUSI Breast Classification",

        modality="ULTRASOUND",

        organ="breast",

        tasks=("classification",),

        input_type=
            "grayscale ultrasound image",

        outputs=(
            "normal",
            "benign",
            "malignant",
        ),

        checkpoint_path=
            "checkpoints/busi/"
            "busi_multitask_unet_best.pt",

        model_kind=
            "multi-task classification head",

        dataset="BUSI",

        priority=20,
    ),



    # =========================
    # MSD Brain Tumor MRI
    # =========================


    ModelSpec(
        model_id="msd_brain_tumor_segmentation",

        name="MSD Brain Tumor MRI Segmentation",

        modality="MRI",

        organ="brain",

        tasks=("segmentation",),

        input_type=
            "3D MRI volume (NIfTI)",

        outputs=(
            "tumor segmentation mask",
            "tumor volume estimation",
        ),

        checkpoint_path=
            "checkpoints/msd_brain_tumor/"
            "msd_brain_tumor_best.pt",

        model_kind=
            "3D U-Net",

        dataset="MSD_Brain_Tumor",

        priority=10,
    ),
    # =========================
    # MSD Pancreas CT
    # =========================

    ModelSpec(
        model_id="msd_pancreas_segmentation",

        name="MSD Pancreas CT Segmentation",

        modality="CT",

        organ="pancreas",

        tasks=("segmentation",),

        input_type=
            "3D CT volume (NIfTI)",

        outputs=(
            "pancreas mask",
            "tumor region mask",
        ),

        checkpoint_path=
            "checkpoints/msd_pancreas_tumor_fast/"
            "pancreas_tumor_fast_best.pt",

        model_kind=
            "3D U-Net",

        dataset="MSD_Pancreas",

        priority=30,
    ),



    # =========================
    # MSD Colon CT
    # =========================

    ModelSpec(
    model_id="msd_colon_tumor_segmentation",
    name="MSD Task10 Colon Tumor CT Segmentation",
    modality="CT",
    organ="colon",
    tasks=("segmentation",),
    input_type="3D abdominal colon CT volume",
    outputs=(
        "colon tumor model-region segmentation mask",
        "tumor volume estimation",
    ),
    checkpoint_path=(
        "checkpoints/msd_colon_tumor/"
        "colon_tumor_best.pt"
    ),
    model_kind="3D U-Net",
    dataset="MSD_Task10_Colon",
    evidence_level="research",
    clinical_validation=False,
    priority=10,
    ),


    # =========================
    # ISIC Skin Lesion
    # =========================

    ModelSpec(
    model_id="isic2016_skin_lesion_segmentation",
    name="ISIC2016 Skin Lesion Segmentation",
    modality="DERMOSCOPY",
    organ="skin",
    tasks=("segmentation",),
    input_type="RGB dermoscopy image",
    outputs=(
        "skin lesion segmentation mask",
        "lesion pixel area",
        "lesion image percentage",
    ),
    checkpoint_path=(
        "checkpoints/isic2016_segmentation/"
        "isic2016_skin_segmentation_best.pt"
    ),
    model_kind="2D U-Net",
    dataset="ISIC2016",
    evidence_level="research",
    clinical_validation=False,
    priority=20,
    ),

    # =========================
    # IRCADb01 Liver CT
    # =========================

    ModelSpec(
        model_id="ircadb01_liver_tumor_segmentation",

        name="IRCADb01 Two-Stage Liver CT Analysis",

        modality="CT",

        organ="liver",

        tasks=("segmentation",),

        input_type=
            "3D liver CT DICOM series",

        outputs=(
            "liver segmentation mask",
            "liver tumor lesion mask",
            "liver volume estimation",
            "tumor volume estimation",
        ),

        checkpoint_path=
            "checkpoints/liver_tumor_stage2/"
            "liver_tumor_stage2_best.pt",

        model_kind=
            "Two-stage 3D U-Net pipeline",

        dataset="IRCADB01",

        evidence_level="research",

        clinical_validation=False,

        priority=10,
    ),


    ModelSpec(
    model_id="msd_pancreas_tumor_segmentation",
    name="MSD Task07 Two-Stage Pancreas CT Analysis",
    modality="CT",
    organ="pancreas",
    tasks=("segmentation",),
    input_type="3D abdominal pancreas CT volume",
    outputs=(
        "pancreas segmentation mask",
        "pancreatic cancer model region mask",
        "pancreas volume estimation",
        "cancer-region volume estimation",
    ),
    checkpoint_path=(
        "checkpoints/msd_pancreas_tumor_fast/"
        "pancreas_tumor_fast_best.pt"
    ),
    model_kind="Two-stage 3D U-Net pipeline",
    dataset="MSD_Pancreas",
    evidence_level="research",
    clinical_validation=False,
    priority=10,
   ),


    # =========================
    # C-NMC Leukemia
    # =========================

    ModelSpec(
    model_id="cnmc2019_all_cell_classifier",
    name="C-NMC 2019 ALL Cell Classifier",
    modality="MICROSCOPY",
    organ="blood",
    tasks=("classification",),
    input_type="RGB microscopy cell image",
    outputs=(
        "ALL-like cell probability",
        "HEM-like cell probability",
        "single-cell classification",
    ),
    checkpoint_path=(
        "checkpoints/cnmc2019/"
        "cnmc2019_classifier_best.pt"
    ),
    model_kind="2D CNN classifier",
    dataset="C-NMC 2019",
    evidence_level="research",
    clinical_validation=False,
    priority=20,
    ),


    # =========================
    # DREAM6 / FlowCAP-II AML
    # =========================

    ModelSpec(
        model_id="flowcap_aml_patient_classifier",
        name="DREAM6 FlowCAP-II AML Patient Classifier",
        modality="FLOW_CYTOMETRY",
        organ="blood",
        tasks=("classification",),
        input_type="8 flow-cytometry CSV tubes",
        outputs=(
            "patient-level DREAM6 model score",
            "8 tube-level model scores",
        ),
        checkpoint_path=(
            "datasets/hematology/flowcap_aml/"
            "Dream6-binary/final-classifier.xml"
        ),
        model_kind="Jstacs DREAM6 pretrained classifier",
        dataset="DREAM6_FlowCAP_II_AML",
        evidence_level="research",
        clinical_validation=False,
        priority=10,
    ),


    # =========================
    # TN3K Thyroid Ultrasound
    # =========================

    ModelSpec(
        model_id="tn3k_thyroid_nodule_segmentation",

        name="TN3K Thyroid Nodule Ultrasound Segmentation",

        modality="ULTRASOUND",

        organ="thyroid",

        tasks=("segmentation",),

        input_type="grayscale thyroid ultrasound image",

        outputs=(
            "thyroid nodule segmentation mask",
            "pixel-based nodule area",
            "pixel-based bounding box",
        ),

        checkpoint_path=(
            "checkpoints/tn3k_segmentation/"
            "tn3k_unet_best.pt"
        ),

        model_kind="2D U-Net",

        dataset="TN3K",

        evidence_level="research",

        clinical_validation=False,

        priority=10,
    ),

)



class ModelRegistry:

    """
    Query model capabilities without importing
    or instantiating models.
    """


    def __init__(
        self,
        specs: Iterable[ModelSpec] = _MODEL_SPECS
    ):

        self._specs = tuple(specs)

        self._by_id = {
            spec.model_id: spec
            for spec in self._specs
        }


        if len(self._by_id) != len(self._specs):

            raise ValueError(
                "Model registry IDs must be unique"
            )



    def list(self):

        return self._specs



    def get_spec(
        self,
        model_id: str
    ):

        return self._by_id.get(model_id)



    def find(
        self,
        *,
        modality: str | None = None,
        organ: str | None = None,
        task: str | None = None,
        available_only: bool = False,

    ):


        normalized_modality = (
            normalize_modality(modality)
            if modality
            else None
        )


        normalized_organ = (
            normalize_organ(organ)
            if organ
            else None
        )


        normalized_task = (
            normalize_task(task)
            if task
            else None
        )


        matches = [

            spec

            for spec in self._specs

            if (
                normalized_modality is None
                or spec.modality == normalized_modality
            )

            and (

                normalized_organ is None
                or spec.organ == normalized_organ

            )

            and (

                normalized_task is None
                or normalized_task in spec.tasks

            )

            and (

                not available_only
                or spec.checkpoint_available

            )

        ]


        return tuple(

            sorted(
                matches,
                key=lambda spec:
                (
                    spec.priority,
                    spec.model_id
                )
            )

        )



    def as_dict(self):

        return {

            spec.model_id:
            spec.to_dict()

            for spec in self._specs

        }




_REGISTRY = ModelRegistry()



MODEL_REGISTRY = (
    _REGISTRY.as_dict()
)



def get_model_registry():

    return _REGISTRY




def get_all_models():

    return _REGISTRY.as_dict()




def get_model(
    model_name: str
):

    spec = _REGISTRY.get_spec(model_name)

    return (
        spec.to_dict()
        if spec
        else None
    )




def get_model_spec(
    model_name: str
):

    return _REGISTRY.get_spec(model_name)
