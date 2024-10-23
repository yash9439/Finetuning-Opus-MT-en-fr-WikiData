# Finetuning-Opus-MT-en-fr-WikiData

```markdown
# English-French Machine Translation Project

## Overview
This project focuses on building and fine-tuning a machine translation model for English-French translation using the Helsinki-NLP/opus-mt-en-fr model from Hugging Face. The project involves dataset selection, extensive data filtering and preprocessing, model training, and performance evaluation using various metrics.

## Dataset Selection

### Source
- **Dataset:** English-French Wiki dataset
- **Source:** [Opus](https://opus.nlpl.eu/Wikipedia/en&fr/v1.0/Wikipedia)

### Initial Sentence Pairs
- **Total Pairs:** 818,302

## Data Filtering and Preprocessing

### Filtering Process
1. **Duplicates Removed:** 803,704 pairs removed
2. **Sentences ≤ 200 Words:** 801,392 pairs retained
3. **Length Ratio ≤ 1.5:** 691,348 pairs retained
4. **Non-printable Characters Removed:** 691,222 pairs retained (126 pairs removed)

### Preprocessing Pipeline
- **Character Cleaning:** Removed non-printable/control characters from both source and target sentences.
- **Normalization:** Applied NFKC normalization and reduced multiple spaces to single spaces.
- **Symbol Removal:** Eliminated unwanted symbols while retaining essential punctuation using regex.

## COMET Scoring and Sampling

### Scoring
- **Method:** Calculated COMET (wmt20-comet-qe-da) scores for 50% of the data (345,611 pairs) for further filtering.

### Language Detection

#### Source Language Distribution
| Language | Count   |
|----------|---------|
| en       | 316,161 |
| fr       | 11,872  |
| de       | 4,039   |
| it       | 2,339   |
| pt       | 818     |
| ca       | 971     |
| nl       | 668     |
| id       | 628     |
| es       | 1,166   |
| ro       | 499     |
| tl       | 1,057   |
| af       | 481     |
| sv       | 528     |
| unknown  | 1,357   |

#### Target Language Distribution
| Language | Count   |
|----------|---------|
| fr       | 298,086 |
| en       | 14,621  |
| de       | 611     |
| it       | 540     |
| ca       | 549     |
| es       | 460     |
| pt       | 142     |
| ro       | 145     |
| nl       | 184     |
| id       | 82      |
| unknown  | 98      |

### Filtered Rows and Sampling
- **Filtered Rows:** 298,086 pairs retained after removing sentences with incorrect language pairs (en-fr).
- **Sampling:** Randomly selected 130K pairs, split into:
  - **Training Set:** 100K pairs
  - **Validation Set:** 15K pairs
  - **Test Set:** 15K pairs

## Model Fine-Tuning

### Model Used
- **Model:** [Helsinki-NLP/opus-mt-en-fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)

> **Note:** The model exhibited overfitting after 10 epochs. Therefore, the checkpoint from Epoch 10 was used for evaluation.

## Performance Evaluation

### Baseline
- **SacreBLEU:** 42.20
- **chrF++:** 65.99
- **COMET (Unbabel/wmt20-comet-qe-da):** 0.395

### Fine-Tuned Model
- **SacreBLEU:** 43.55
- **chrF++:** 69.16
- **COMET (Unbabel/wmt20-comet-qe-da):** 0.566

