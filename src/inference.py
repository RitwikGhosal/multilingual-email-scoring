import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from model import load_multitask_checkpoint


REQUIRED_COLUMNS = {
    "questionStatement",
    "text",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Predict grammar, content, and "
            "CEFR scores for multilingual emails."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input .csv, .xlsx, or .xls file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .csv or .xlsx file.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Inference configuration file. "
            "Defaults to inference_config.json "
            "in the package root."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return parser.parse_args()


def resolve_path(config_dir, value):
    path = Path(value)

    if path.is_absolute():
        return path

    return (
        config_dir / path
    ).resolve()


def load_input(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(
            "Input must be .csv, .xlsx, "
            "or .xls"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(frame) == 0:
        raise ValueError(
            "The input file contains no rows."
        )

    if "questionID" not in frame.columns:
        frame.insert(
            0,
            "questionID",
            np.arange(len(frame)),
        )

    return frame


def normalize_text(value):
    if pd.isna(value):
        value = ""

    return unicodedata.normalize(
        "NFC",
        str(value),
    )


def select_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is "
            "not available."
        )

    return torch.device(requested)


def load_calibration(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(file)

    cut_points = np.asarray(
        calibration["cut_points"],
        dtype=np.float32,
    )

    if cut_points.shape != (5,):
        raise ValueError(
            "CEFR calibration must contain "
            "exactly five cut-points."
        )

    if not np.all(
        np.diff(cut_points) > 0
    ):
        raise ValueError(
            "CEFR cut-points are not "
            "strictly increasing."
        )

    return cut_points


def save_output(frame, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(
            path,
            index=False,
        )
    elif suffix == ".xlsx":
        frame.to_excel(
            path,
            index=False,
        )
    else:
        raise ValueError(
            "Output must be .csv or .xlsx"
        )


@torch.inference_mode()
def predict(
    frame,
    tokenizer,
    grammar_model,
    content_cefr_model,
    cut_points,
    cefr_labels,
    batch_size,
    max_length,
    device,
):
    grammar_predictions = []
    content_predictions = []
    cefr_soft_scores = []
    cefr_uncalibrated_ids = []

    questions = (
        frame["questionStatement"]
        .map(normalize_text)
        .tolist()
    )

    responses = (
        frame["text"]
        .map(normalize_text)
        .tolist()
    )

    for start in tqdm(
        range(0, len(frame), batch_size),
        desc="Running inference",
    ):
        stop = min(
            start + batch_size,
            len(frame),
        )

        batch = tokenizer(
            questions[start:stop],
            responses[start:stop],
            truncation=True,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        model_batch = {
            "input_ids":
                batch["input_ids"].to(
                    device,
                    non_blocking=True,
                ),
            "attention_mask":
                batch["attention_mask"].to(
                    device,
                    non_blocking=True,
                ),
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            grammar_output = grammar_model(
                **model_batch
            )

            content_cefr_output = (
                content_cefr_model(
                    **model_batch
                )
            )

        grammar_predictions.extend(
            grammar_output[
                "grammar_predictions"
            ]
            .float()
            .cpu()
            .numpy()
        )

        content_predictions.extend(
            content_cefr_output[
                "content_predictions"
            ]
            .float()
            .cpu()
            .numpy()
        )

        cefr_soft_scores.extend(
            content_cefr_output[
                "cefr_soft_scores"
            ]
            .float()
            .cpu()
            .numpy()
        )

        cefr_uncalibrated_ids.extend(
            content_cefr_output[
                "cefr_uncalibrated_ids"
            ]
            .cpu()
            .numpy()
        )

    grammar_predictions = np.asarray(
        grammar_predictions,
        dtype=np.float32,
    )

    content_predictions = np.asarray(
        content_predictions,
        dtype=np.float32,
    )

    cefr_soft_scores = np.asarray(
        cefr_soft_scores,
        dtype=np.float32,
    )

    calibrated_ids = (
        cefr_soft_scores[:, None]
        > cut_points[None, :]
    ).sum(axis=1)

    calibrated_labels = [
        cefr_labels[int(class_id)]
        for class_id in calibrated_ids
    ]

    output = frame.copy()

    output["grammar_prediction"] = (
        grammar_predictions
    )

    output["content_prediction"] = (
        content_predictions
    )

    output["cefr_soft_score"] = (
        cefr_soft_scores
    )

    output[
        "cefr_uncalibrated_id"
    ] = np.asarray(
        cefr_uncalibrated_ids,
        dtype=np.int64,
    )

    output[
        "cefr_prediction_id"
    ] = calibrated_ids.astype(
        np.int64
    )

    output["cefr_prediction"] = (
        calibrated_labels
    )

    return output


def main():
    arguments = parse_arguments()

    script_path = Path(__file__).resolve()
    package_dir = script_path.parent.parent

    config_path = (
        Path(arguments.config).resolve()
        if arguments.config
        else package_dir
        / "inference_config.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    config_dir = config_path.parent
    device = select_device(
        arguments.device
    )

    batch_size = (
        arguments.batch_size
        if arguments.batch_size is not None
        else int(config["batch_size"])
    )

    tokenizer_dir = resolve_path(
        config_dir,
        config["tokenizer_dir"],
    )

    grammar_checkpoint = resolve_path(
        config_dir,
        config[
            "grammar_model"
        ]["checkpoint_dir"],
    )

    content_checkpoint = resolve_path(
        config_dir,
        config[
            "content_cefr_model"
        ]["checkpoint_dir"],
    )

    calibration_path = (
        content_checkpoint
        / config[
            "content_cefr_model"
        ]["calibration_file"]
    )

    print("Device:", device)
    print(
        "Grammar checkpoint:",
        grammar_checkpoint,
    )
    print(
        "Content/CEFR checkpoint:",
        content_checkpoint,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
        )
    )

    grammar_model = (
        load_multitask_checkpoint(
            base_model_name=(
                config["base_model"]
            ),
            checkpoint_dir=(
                grammar_checkpoint
            ),
            pooling=config[
                "grammar_model"
            ]["pooling"],
            device=device,
            projection_size=int(
                config["projection_size"]
            ),
            dropout=float(
                config["dropout"]
            ),
            heads_filename=config[
                "grammar_model"
            ]["heads_filename"],
        )
    )

    content_cefr_model = (
        load_multitask_checkpoint(
            base_model_name=(
                config["base_model"]
            ),
            checkpoint_dir=(
                content_checkpoint
            ),
            pooling=config[
                "content_cefr_model"
            ]["pooling"],
            device=device,
            projection_size=int(
                config["projection_size"]
            ),
            dropout=float(
                config["dropout"]
            ),
            heads_filename=config[
                "content_cefr_model"
            ]["heads_filename"],
        )
    )

    cut_points = load_calibration(
        calibration_path
    )

    input_frame = load_input(
        arguments.input
    )

    output_frame = predict(
        frame=input_frame,
        tokenizer=tokenizer,
        grammar_model=grammar_model,
        content_cefr_model=(
            content_cefr_model
        ),
        cut_points=cut_points,
        cefr_labels=config[
            "cefr_labels"
        ],
        batch_size=batch_size,
        max_length=int(
            config["max_length"]
        ),
        device=device,
    )

    save_output(
        output_frame,
        arguments.output,
    )

    print(
        "Rows processed:",
        len(output_frame),
    )
    print(
        "Saved predictions:",
        Path(arguments.output).resolve(),
    )


if __name__ == "__main__":
    main()
