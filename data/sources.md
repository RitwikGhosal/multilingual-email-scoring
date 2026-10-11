# Data Sources

The prepared dataset combines controlled multilingual email examples with publicly available CEFR learner texts.

## Controlled data

| Source | Rows | Description |
|---|---:|---|
| `synthetic_curated_v2` | 1800 | Controlled grammar and content variants derived from the multilingual email scenario bank. |
| `synthetic_rubric_v2` | 200 | Separately constructed C1 and C2 email variants. |

The controlled examples were created for this project with LLM-assisted drafting and manual inspection. They were not professionally translated or independently rated by native-language CEFR assessors.

## Authentic learner data

| Dataset | Rows | Recorded licence | Source page |
|---|---:|---|---|
| `UniversalCEFR/elg_cefr_nl` | 645 | CC BY-NC-SA 4.0 | https://huggingface.co/datasets/UniversalCEFR/elg_cefr_nl |
| `UniversalCEFR/merlin_it` | 600 | CC BY-SA 4.0 | https://huggingface.co/datasets/UniversalCEFR/merlin_it |
| `UniversalCEFR/cople2_pt` | 446 | CC BY-SA-NC 4.0 | https://huggingface.co/datasets/UniversalCEFR/cople2_pt |
| `UniversalCEFR/peapl2_pt` | 309 | CC BY SA NC 4.0 | https://huggingface.co/datasets/UniversalCEFR/peapl2_pt |

## Redistribution note

The original corpus documentation and licences remain authoritative. Users should review the linked dataset pages before redistributing or using the authentic texts outside the permitted terms.

The authentic rows provide CEFR supervision only. Their grammar and content labels are missing and are masked during multitask training.
