import json

from urban_skin.data.audit import audit_dataset


def test_audit_counts_samples_and_labels(tmp_path) -> None:
    dataset = tmp_path / "demo"
    dataset.mkdir()
    (dataset / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "tags": ["train"],
                        "od_ground_truth": {
                            "detections": [{"label": "Crack"}, {"label": "Spalling"}]
                        },
                    },
                    {"tags": ["test"], "od_ground_truth": {"detections": []}},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert audit_dataset(dataset) == {
        "dataset": "demo",
        "samples": 2,
        "splits": {"test": 1, "train": 1},
        "images_with_labels": 1,
        "annotations": 2,
        "classes": {"Crack": 1, "Spalling": 1},
    }
