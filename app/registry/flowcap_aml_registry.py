FLOWCAP_AML_MODEL = {
    "id": "flowcap_aml_patient_classifier",

    "name": "DREAM6 FlowCAP-II AML Classifier",

    "domain": "hematology",

    "modality": "FLOW_CYTOMETRY",

    "anatomy": "blood_bone_marrow",

    "task": "patient_level_classification",

    "input": {
        "tube_count": 8,
        "format": "CSV",
        "channels": [
            "FS Lin",
            "SS Log",
            "FL1 Log",
            "FL2 Log",
            "FL3 Log",
            "FL4 Log",
            "FL5 Log",
        ],
    },

    "model": {
        "framework": "Jstacs",
        "classifier": "Dream6C4",
        "pretrained": True,
        "checkpoint": (
            "datasets/hematology/flowcap_aml/"
            "Dream6-binary/final-classifier.xml"
        ),
    },

    "dataset": {
        "name": "DREAM6 / FlowCAP-II AML",
        "subjects": 359,
        "files": 2872,
        "tubes_per_subject": 8,
    },

    "clinical_status": {
        "research_only": True,
        "clinically_validated": False,
        "diagnostic_use": False,
    },
}