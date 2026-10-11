# Multilingual Email Scoring

A multilingual system for scoring email responses on three dimensions:

- **Grammar:** 0 to 5
- **Content coverage:** 0 to 4
- **CEFR proficiency:** A1 to C2

The supported languages are Dutch, Greek, Italian, Polish, and Portuguese. The model receives both the writing prompt and the email response.

The system uses XLM-RoBERTa-base with LoRA adapters and task-specific prediction heads. The default inference mode combines the strongest grammar checkpoint with the strongest content and CEFR checkpoint.

For full implementation details, see:

 <!--- [Short approach summary](docs/approach_summary.pdf)
- [Detailed technical report](docs/technical_report.pdf) -->

## Repository Structure

```text
.
├── data/
│   ├── multilingual_training_data.csv
│   ├── multilingual_training_data.pkl
│   ├── data_schema.md
│   └── sources.md
├── docs/
│   ├── approach_summary.pdf
│   └── technical_report.pdf
├── examples/
│   ├── sample_input.csv
│   ├── sample_predictions.csv
│   ├── sample_predictions_cls.csv
│   └── sample_predictions_mean.csv
├── models/
│   ├── hybrid_cls/
│   └── hybrid_mean/
├── notebooks/
├── src/
│   ├── inference.py
│   └── model.py
├── inference_config.json
├── install.sh
├── requirements.txt
└── requirements-lock.txt
```

## Installation

Python 3.11 or later is recommended. A CUDA GPU is helpful but not required.

```bash
bash install.sh
```

The installation script removes an optional, incompatible TorchAO package that may be preinstalled in some Colab environments and then installs the required dependencies.

The base `xlm-roberta-base` weights are downloaded from Hugging Face when inference is run for the first time. The repository contains the trained LoRA adapters, prediction heads, tokenizer configuration, and CEFR calibration files.

## Input Format

The inference script accepts CSV, XLSX, or XLS files.

Two columns are required:

| Column | Description |
|---|---|
| `questionStatement` | Email-writing prompt |
| `text` | Email response to score |

An optional `questionID` column can be supplied. If it is absent, sequential identifiers are created automatically.

Example:

```csv
questionID,questionStatement,text
example_01,"Write an email requesting a meeting change.","Good morning, I would like to move our meeting..."
```

A complete example is available at [`examples/sample_input.csv`](examples/sample_input.csv).

## Running Inference

The default mode is the task-specific ensemble:

```bash
python src/inference.py \
    --input examples/sample_input.csv \
    --output predictions.csv
```

The mode can also be stated explicitly:

```bash
python src/inference.py \
    --input examples/sample_input.csv \
    --output predictions.csv \
    --mode ensemble
```

### Available modes

| Mode | Grammar | Content | CEFR |
|---|---|---|---|
| `ensemble` | First-token model | Mean-pooling model | Mean-pooling model |
| `cls` | First-token model | First-token model | First-token model |
| `mean` | Mean-pooling model | Mean-pooling model | Mean-pooling model |

The default `ensemble` mode gave the strongest task-specific combination on internal validation.

Single-model inference requires less memory:

```bash
python src/inference.py \
    --input examples/sample_input.csv \
    --output predictions_cls.csv \
    --mode cls
```

or:

```bash
python src/inference.py \
    --input examples/sample_input.csv \
    --output predictions_mean.csv \
    --mode mean
```

Additional options include:

```text
--config PATH
--batch-size INTEGER
--device auto|cuda|cpu
```

Run the following command for the complete interface:

```bash
python src/inference.py --help
```

## Output Format

The output preserves the original rows and adds:

| Column | Description |
|---|---|
| `inference_mode` | Selected inference mode |
| `grammar_prediction` | Continuous grammar score in `[0, 5]` |
| `content_prediction` | Continuous content score in `[0, 4]` |
| `cefr_soft_score` | Continuous ordinal CEFR score |
| `cefr_uncalibrated_id` | CEFR class before calibration |
| `cefr_prediction_id` | Calibrated CEFR class from 0 to 5 |
| `cefr_prediction` | Calibrated label from A1 to C2 |

Example outputs for all three modes are provided in the `examples/` directory.

## Model

The shared encoder is `xlm-roberta-base`. LoRA adapters are applied to the query and value projections with:

- Rank: 8
- LoRA scaling factor: 16
- LoRA dropout: 0.10
- Projection dimension: 256

Grammar and content use separate bounded regression heads. A sigmoid restricts their predictions to the required rubric ranges.

CEFR uses an ordinal prediction head with five learned, strictly ordered thresholds. Passing zero thresholds corresponds to A1, while passing all five corresponds to C2.

Two pooling strategies were evaluated:

- Representation of the first XLM-R token
- Attention-mask-aware mean pooling over token representations

## Training Data

The prepared dataset contains 4,000 rows:

| Source | Rows |
|---|---:|
| Controlled multilingual data | 2,000 |
| UniversalCEFR ELG CEFR Dutch | 645 |
| UniversalCEFR MERLIN Italian | 600 |
| UniversalCEFR COPLE2 Portuguese | 446 |
| UniversalCEFR PEAPL2 Portuguese | 309 |

The split contains 3,200 training and 800 validation rows.

The controlled corpus was constructed from 40 practical email scenarios represented in all five languages. Each response was divided into functional components, including purpose, context, action, supporting information, and confirmation.

Content variants were created by retaining or omitting atomic task details. Grammar variants were generated separately using typed language-specific transformations such as agreement errors, incorrect articles or prepositions, verb-form errors, missing diacritics, sentence fragments, and limited spelling noise.

Authentic learner corpora were added to improve linguistic realism. Exact CEFR labels were retained where available. Labels such as A1-A2, B1-B2, or C1-C2 were trained as intervals rather than being converted to arbitrary exact classes.

The full schema is documented in [`data/data_schema.md`](data/data_schema.md). Dataset sources and licence information are listed in [`data/sources.md`](data/sources.md).

## Training Strategy

The model was fine-tuned using:

- AdamW
- Eight epochs for the final hybrid experiments
- LoRA learning rate: `2e-4`
- Prediction-head learning rate: `5e-4`
- Weight decay: `0.01`
- Ten percent linear warmup
- Linear learning-rate decay
- Mixed-precision training
- Gradient clipping at `1.0`

The total multitask loss is the sum of:

1. Normalized mean squared error for grammar
2. Normalized mean squared error for content
3. Ordinal binary cross-entropy for CEFR

Missing task labels are masked. Interval CEFR examples supervise only threshold decisions that are unambiguous for the complete interval.

The notebooks in `notebooks/` contain the data processing, training, evaluation, checkpoint saving, and calibration workflow. The detailed technical report explains each stage.

## Internal Validation Results

| Metric | First-token model | Mean-pooling model |
|---|---:|---:|
| Grammar Pearson | **0.9208** | 0.9063 |
| Content Pearson | 0.9710 | **0.9767** |
| CEFR soft Pearson | **0.8702** | 0.8580 |
| CEFR exact accuracy | 0.5977 | 0.6256 |
| Calibrated CEFR accuracy | 0.6674 | **0.6767** |

The first-token model was selected for grammar. The mean-pooling model was selected for content and calibrated CEFR prediction.

CEFR calibration changes only the conversion from the continuous ordinal score to a discrete class. It does not update the trained model parameters.

## Reproducing the Models

1. Install the dependencies:

   ```bash
   bash install.sh
   ```

2. Use the prepared dataset:

   ```text
   data/multilingual_training_data.pkl
   ```

3. Run the clean training notebooks in `notebooks/`.

4. Save the resulting LoRA adapters, prediction heads, tokenizer files, and CEFR calibration under the corresponding directory in `models/`.

The inference configuration is stored in `inference_config.json`.

## Verification

The complete GitHub repository archive was tested in a fresh Colab T4 runtime. The test included:

- Dependency installation
- Loading both checkpoints without notebook state
- Running `ensemble`, `cls`, and `mean` modes
- Preserving row order
- Checking for missing predictions
- Verifying grammar and content ranges
- Verifying valid CEFR labels

All three inference modes passed.

## Limitations

The controlled corpus provides precise rubric supervision but cannot cover every naturally occurring learner error. The authentic corpora improve realism, although their language and CEFR distributions are uneven and they do not contain grammar or content annotations.

The internal validation metrics should therefore be interpreted as measurements on the constructed validation distribution, not as guarantees across every writing domain or learner population.

## AI Assistance Disclosure

Large language model assistance was used for brainstorming, code scaffolding and debugging, multilingual drafting, and documentation editing. The author was responsible for the final design choices, implementation, execution, validation, and interpretation.
