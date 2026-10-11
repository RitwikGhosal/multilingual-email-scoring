# Prepared Training Data Schema

The prepared hybrid dataset contains 4,000 rows: 3,200 training rows
and 800 validation rows. It combines 2,000 controlled multilingual
email examples with 2,000 authentic CEFR-labelled learner texts.

The pickle file preserves Python lists and dictionaries. In the CSV
file, structured values such as error records and component lists are
stored as string representations.

Missing task labels are converted to `-100` when the training notebook
constructs model tensors.

| Column | Description |
|---|---|
| `sample_id` | Unique identifier for the generated or imported example. |
| `scenario_id` | Identifier of the controlled email scenario. Missing for authentic corpus rows. |
| `language` | Italian, Dutch, Greek, Polish, or Portuguese. |
| `category` | Broad scenario category for controlled examples. |
| `sub_category` | More specific controlled scenario category. |
| `split` | `train` or `validation`. |
| `register` | Formal or semi-formal email register. |
| `question_style` | Surface form used for the writing prompt. |
| `questionStatement` | Email-writing prompt. Empty for authentic corpus rows without compatible prompts. |
| `text` | Candidate email or authentic learner response used as model input. |
| `content` | Content score from 0 to 4. Missing for authentic CEFR-only rows. |
| `grammar` | Grammar score from 0 to 5. Missing for authentic CEFR-only rows. |
| `cefr` | Exact CEFR label when available. |
| `components_present` | Task components retained in a controlled response. |
| `components_omitted` | Task components removed from a controlled response. |
| `details_present` | Atomic task-detail identifiers retained in the response. |
| `details_omitted` | Atomic task-detail identifiers omitted from the response. |
| `body_sentence_count` | Number of body sentences in a controlled email. |
| `content_generation` | Content-generation method or score-specific policy. |
| `source` | Synthetic source identifier or public dataset identifier. |
| `text_clean` | Clean response before grammatical perturbation. |
| `parent_sample_id` | Identifier of the clean parent response. |
| `grammar_generation` | Grammar-generation policy used for the row. |
| `grammar_errors` | Recorded error operations, including type, severity, before text, after text, and method. |
| `grammar_error_count` | Number of recorded grammatical or orthographic operations. |
| `grammar_severity_total` | Sum of recorded error severities. |
| `protected_tokens_preserved` | Whether identifiers, numbers, placeholders, and other protected tokens remained unchanged. |
| `cefr_source` | Method or source used to assign CEFR supervision. |
| `upper_body_unit_count` | Number of body units in a generated C1 or C2 response. |
| `word_count` | Word count for controlled upper-level examples. |
| `upper_joiner` | Cohesive construction used in an upper-level response. |
| `upper_development_clause` | Development or nuance clause used in C1/C2 generation. |
| `label_origin` | Whether the CEFR annotation is synthetic, exact external, or interval-based. |
| `source_id` | Original identifier of an authentic corpus response. |
| `cefr_min` | Lower CEFR bound encoded from 0 for A1 to 5 for C2. |
| `cefr_max` | Upper CEFR bound encoded from 0 for A1 to 5 for C2. |
| `cefr_is_exact` | Whether the minimum and maximum CEFR levels are identical. |
| `cefr_band` | Original exact or adjacent CEFR band, such as `B1` or `B1/B2`. |
| `external_category` | Category supplied by the external corpus, when available. |
| `external_format` | Original format or task type of the external response. |
| `external_license` | Licence metadata supplied for the external source. |
| `external_title` | Original corpus title or response title. |
| `external_dataset_id` | Public dataset identifier used to retrieve the source. |
| `word_count_external` | Word count recorded for an external learner response. |

## Label ranges

- Grammar: 0 to 5
- Content: 0 to 4
- CEFR: A1, A2, B1, B2, C1, C2
- CEFR integer mapping: A1=0, A2=1, B1=2, B2=3, C1=4, C2=5

Rows without a valid task label are masked during training and do not
contribute to that task's loss.
