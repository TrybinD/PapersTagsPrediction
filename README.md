# PapersTagsPrediction
Repository for the NLP course project

The main goal of this project is to learn how to predict tags and keywords from the text. This can be used to build search engines, recommendation systems, and so on.

[Extreme Multi-label Text Classification for
Habr posts](https://github.com/TrybinD/PapersTagsPrediction/blob/main/PapersTagPrediction.pdf)

## Installation

The prerequisites are specified in py project.toml

To install dependencies run:
```
poetry install
```

## Datasets

The our collected dataset is available at the [link](https://drive.google.com/file/d/1c3akpSM7RdPsvuwcggtO-c3TTXFRVeXg/view?usp=sharing)

## Run

To run Habr parsing process look at `notebooks/parse_habr.ipynb`

To run models from Releted Works look at `notebooks/related_works.ipynb`

To run LDA-based models look at `notebooks/lda_embeddings.ipynb`

To run RuBERT-based models look at `notebooks/rubert.ipynb`

## Baselines

The codes for the baseline models are adapted from the following repositories: [XML-CNN](https://github.com/castorini/hedwig), [MeSHProbeNet](https://github.com/XunGuangxu/MeSHProbeNet), [AttentionXML](https://github.com/yourh/AttentionXML) and [CorNet](https://github.com/XunGuangxu/CorNet).

