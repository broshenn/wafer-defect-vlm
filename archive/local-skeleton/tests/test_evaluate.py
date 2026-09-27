from wafer_vlm.evaluate import predicted_label, robustness_metrics, response_text, strip_thinking


def test_predicted_label_flags_off_vocabulary_answers() -> None:
    assert predicted_label("Scratch") == "Scratch"
    assert predicted_label("scratch") == "Scratch"
    assert predicted_label("none") == "none"
    # Answers outside the requested vocabulary must not look like ordinary errors.
    assert predicted_label("Particle") == "INVALID"
    assert predicted_label("") == "INVALID"


def test_empty_think_block_is_removed_before_parsing() -> None:
    assert strip_thinking("<think>  </think>  Scratch") == "Scratch"
    assert strip_thinking("<think>\nlong reasoning\n</think>\n```json\n{}\n```") == "```json\n{}\n```"


def test_think_block_is_case_insensitive_and_multiline() -> None:
    raw = "<THINK>\nstep 1\nstep 2\n</THINK >\nEdge_Ring"
    assert strip_thinking(raw) == "Edge_Ring"


def test_response_text_strips_thinking_from_swift_rows() -> None:
    row = {"response": "<think>  </think>  Center", "messages": [], "images": []}
    assert response_text(row) == "Center"


def test_response_text_survives_plain_answers() -> None:
    assert response_text({"response": "Donut"}) == "Donut"
    assert response_text({"choices": [{"message": {"content": "Loc"}}]}) == "Loc"
    assert (
        response_text({"messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "Near_full"}]})
        == "Near_full"
    )


VARIANTS = {
    "s1__resize_224": {"variant_id": "s1__resize_224", "sample_id": "s1",
                       "perturbation": "resize_224", "failure_type": "Scratch"},
    "s1__rotate_90_ccw": {"variant_id": "s1__rotate_90_ccw", "sample_id": "s1",
                          "perturbation": "rotate_90_ccw", "failure_type": "Scratch"},
}


def test_robustness_scores_perturbations_and_flips() -> None:
    rows = [
        {"sample_id": "s1", "task": "classification", "response": "Scratch"},
        {"sample_id": "s1__resize_224", "task": "robustness_classification", "response": "Scratch"},
        # Rotation flips the answer, which is exactly what this metric must catch.
        {"sample_id": "s1__rotate_90_ccw", "task": "robustness_classification", "response": "Donut"},
    ]
    report = robustness_metrics(rows, VARIANTS)
    assert report["samples"] == 2
    assert report["accuracy"] == 0.5
    assert report["accuracy_per_perturbation"]["resize_224"] == 1.0
    assert report["accuracy_per_perturbation"]["rotate_90_ccw"] == 0.0
    assert report["flip_rate_vs_clean"] == 0.5
    assert report["invalid_predictions"] == 0


def test_robustness_ignores_unknown_variants_and_counts_invalid() -> None:
    rows = [
        {"sample_id": "ghost", "task": "robustness_classification", "response": "Scratch"},
        {"sample_id": "s1__resize_224", "task": "robustness_classification", "response": "nonsense"},
    ]
    report = robustness_metrics(rows, VARIANTS)
    assert report["samples"] == 1          # the unknown variant is skipped
    assert report["invalid_predictions"] == 1
    assert report["accuracy"] == 0.0
