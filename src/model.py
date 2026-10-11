import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel


class OrderedCEFRHead(nn.Module):
    """Ordinal CEFR head with five ordered thresholds."""

    def __init__(self, input_size):
        super().__init__()

        self.score = nn.Linear(input_size, 1)

        self.base_threshold = nn.Parameter(
            torch.tensor(-2.0)
        )

        inverse_softplus_one = math.log(
            math.exp(1.0) - 1.0
        )

        self.raw_steps = nn.Parameter(
            torch.full(
                (4,),
                inverse_softplus_one,
            )
        )

    def get_thresholds(self):
        positive_steps = F.softplus(
            self.raw_steps
        )

        remaining_thresholds = (
            self.base_threshold
            + torch.cumsum(
                positive_steps,
                dim=0,
            )
        )

        return torch.cat(
            [
                self.base_threshold.reshape(1),
                remaining_thresholds,
            ]
        )

    def forward(self, hidden_states):
        score = self.score(hidden_states)
        thresholds = self.get_thresholds()

        return (
            score
            - thresholds.unsqueeze(0)
        )


class XLMRMultitaskModel(nn.Module):

    def __init__(
        self,
        encoder,
        hidden_size,
        pooling,
        projection_size=256,
        dropout=0.10,
    ):
        super().__init__()

        if pooling not in {
            "first_token",
            "masked_mean",
        }:
            raise ValueError(
                f"Unsupported pooling: {pooling}"
            )

        self.encoder = encoder
        self.pooling = pooling

        self.shared_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                projection_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.grammar_head = nn.Linear(
            projection_size,
            1,
        )

        self.content_head = nn.Linear(
            projection_size,
            1,
        )

        self.cefr_head = OrderedCEFRHead(
            projection_size
        )

    def pool(
        self,
        token_embeddings,
        attention_mask,
    ):
        if self.pooling == "first_token":
            return token_embeddings[:, 0]

        expanded_mask = (
            attention_mask
            .unsqueeze(-1)
            .to(token_embeddings.dtype)
        )

        numerator = (
            token_embeddings
            * expanded_mask
        ).sum(dim=1)

        denominator = (
            expanded_mask
            .sum(dim=1)
            .clamp(min=1e-9)
        )

        return numerator / denominator

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = (
            encoder_output.last_hidden_state
        )

        pooled_output = self.pool(
            token_embeddings,
            attention_mask,
        )

        features = self.shared_projection(
            pooled_output
        )

        grammar_predictions = (
            torch.sigmoid(
                self.grammar_head(features)
            )
            .squeeze(-1)
            * 5.0
        )

        content_predictions = (
            torch.sigmoid(
                self.content_head(features)
            )
            .squeeze(-1)
            * 4.0
        )

        cefr_logits = self.cefr_head(
            features
        )

        cefr_probabilities = torch.sigmoid(
            cefr_logits
        )

        return {
            "grammar_predictions":
                grammar_predictions,
            "content_predictions":
                content_predictions,
            "cefr_logits":
                cefr_logits,
            "cefr_probabilities":
                cefr_probabilities,
            "cefr_soft_scores":
                cefr_probabilities.sum(dim=1),
            "cefr_uncalibrated_ids":
                (cefr_probabilities > 0.5)
                .sum(dim=1),
        }


def load_multitask_checkpoint(
    base_model_name,
    checkpoint_dir,
    pooling,
    device,
    projection_size=256,
    dropout=0.10,
    heads_filename="prediction_heads_epoch8.pt",
):
    checkpoint_dir = Path(checkpoint_dir)

    adapter_dir = (
        checkpoint_dir / "lora_adapter"
    )

    heads_path = (
        checkpoint_dir / heads_filename
    )

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Missing adapter: {adapter_dir}"
        )

    if not heads_path.exists():
        raise FileNotFoundError(
            f"Missing prediction heads: "
            f"{heads_path}"
        )

    base_encoder = AutoModel.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
    )

    hidden_size = (
        base_encoder.config.hidden_size
    )

    encoder = PeftModel.from_pretrained(
        base_encoder,
        str(adapter_dir),
        is_trainable=False,
    )

    model = XLMRMultitaskModel(
        encoder=encoder,
        hidden_size=hidden_size,
        pooling=pooling,
        projection_size=projection_size,
        dropout=dropout,
    )

    try:
        saved_heads = torch.load(
            heads_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        saved_heads = torch.load(
            heads_path,
            map_location="cpu",
        )

    required_sections = {
        "shared_projection",
        "grammar_head",
        "content_head",
        "cefr_head",
    }

    missing_sections = (
        required_sections
        - set(saved_heads)
    )

    if missing_sections:
        raise KeyError(
            "Missing head sections: "
            f"{sorted(missing_sections)}"
        )

    model.shared_projection.load_state_dict(
        saved_heads["shared_projection"],
        strict=True,
    )

    model.grammar_head.load_state_dict(
        saved_heads["grammar_head"],
        strict=True,
    )

    model.content_head.load_state_dict(
        saved_heads["content_head"],
        strict=True,
    )

    model.cefr_head.load_state_dict(
        saved_heads["cefr_head"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model
