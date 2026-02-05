<p align="center">
  <h1 align="center">EmoDynamiX Telugu</h1>
  <h3 align="center">Low-Resource Adaptation of EmoDynamiX for Strategy Prediction in Telugu Emotional Support Dialogues</h3>
</p>

This repository adapts EmoDynamiX to Telugu by switching the encoder to XLM-R, adding a Telugu ERC model for mixed-emotion signals, and supporting faster training via precomputed discourse parses and ERC logits. The goal is strategy prediction for emotional support dialogues in a low-resource setting.

## What is included

- XLM-R based heterogeneous-graph model for Telugu strategy prediction.
- Telugu ESConv dataset and preprocessing utilities.
- Training, evaluation, and inference scripts for Telugu.
- Optional acceleration path with cached discourse parses and ERC logits.

## Repository layout

- Core training entrypoint: [main.py](main.py)
- Telugu model implementation: [modules/roberta/model_telugu.py](modules/roberta/model_telugu.py)
- Telugu inference helper: [infer_telugu_custom.py](infer_telugu_custom.py)
- Comprehensive evaluation: [evaluate_telugu.py](evaluate_telugu.py)
- Telugu preprocessing: [make_telugu_preprocessed.py](make_telugu_preprocessed.py)

## Setup

Choose one of the following environments.

Conda (recommended):

```powershell
conda env create -f environment.yaml
conda activate emodynamix
```

Pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Telugu ESConv data is provided in [esconv_telugu/ESConv_Telugu.json](esconv_telugu/ESConv_Telugu.json). The strategy labels come from [data/esconv/strategies.json](data/esconv/strategies.json).

If you want to regenerate or preprocess datasets, use the scripts in [make_telugu_preprocessed.py](make_telugu_preprocessed.py) and [preprocess.py](preprocess.py).

## Training (Telugu)

Train the Telugu model on ESConv Telugu:

```powershell
./train_telugu.ps1 -Model xlmr-hg-telugu -Dataset esconv-telugu -BatchSize 8 -Epochs 10 -TotalSteps 5000 -HgDim 512 -TeluguErcPath telugu_erc_xlmroberta_trained_v2
```

Key arguments:

- `--model xlmr-hg-telugu` selects the Telugu XLM-R graph model.
- `--dataset esconv-telugu` loads the Telugu dataset.
- `--telugu_erc_path` points to the Telugu ERC checkpoint used for emotion mixing.

## Faster training with cached logits and parses

Precompute ERC logits and discourse parses, then train in light mode:

```powershell
python make_telugu_preprocessed.py --telugu_erc_path telugu_erc_xlmroberta_trained_v2
python main.py --mode train --model xlmr-hg-telugu --dataset esconv-telugu-preprocessed --lightmode 1 --total_steps 5000 --total_epochs 10 --batch_size 8 --hg_dim 512 --telugu_erc_path telugu_erc_xlmroberta_trained_v2
```

Notes:

- Light mode skips on-the-fly parsing and uses cached `parsed_dialogue` and `erc_logits`.
- The preprocessor uses the English discourse parser; this is a speed optimization, not a Telugu parser.

## Evaluation

Evaluate a Telugu checkpoint and generate a confusion matrix and ablations:

```powershell
python evaluate_telugu.py --checkpoint_dir xlmr-hg-telugu-esconv-telugu-checkpoints --checkpoint_step 5000
```

The results are saved to JSON and figures under the working directory.

## Inference (custom dialogue)

Run prediction for a custom Telugu dialogue JSON:

```powershell
python infer_telugu_custom.py --checkpoint_dir xlmr-hg-telugu-esconv-telugu-checkpoints --checkpoint_step 5000 --dialogue_path my_dialogue.json
```

The JSON format is a list of turns or an object with `dialog`. Each turn supports `speaker`, `content`, and optional `annotation.strategy` and `annotation.feedback`. Example is included in [infer_telugu_custom.py](infer_telugu_custom.py).

## Model summary

The Telugu model follows EmoDynamiX but replaces the encoder with XLM-R and injects Telugu ERC signals. The model constructs a discourse-aware graph of dialogue turns and predicts the next supporter strategy. Mixed-emotion prototypes are combined using ERC logits; in light mode these logits can be precomputed to speed up training.

## Citation

If you use this adaptation, please cite the original EmoDynamiX paper:

```bibtex
@inproceedings{wan-etal-2025-emodynamix,
  author       = {Chenwei Wan and
                  Matthieu Labeau and
                  Chlo{\'{e}} Clavel},
  editor       = {Luis Chiruzzo and
                  Alan Ritter and
                  Lu Wang},
  title        = {EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling
                  MiXed Emotions and Discourse Dynamics},
  booktitle    = {Proceedings of the 2025 Conference of the Nations of the Americas
                  Chapter of the Association for Computational Linguistics: Human Language
                  Technologies, {NAACL} 2025 - Volume 1: Long Papers, Albuquerque, New
                  Mexico, USA, April 29 - May 4, 2025},
  pages        = {1678--1695},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.naacl-long.81},
  doi          = {10.18653/V1/2025.NAACL-LONG.81}
}
```
