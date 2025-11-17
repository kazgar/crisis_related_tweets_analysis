# Tweet Classification Project

This project is focused on classifying tweets related to humanitarian crises. It provides pipelines for **multiclass classification** (tweet content categories) and **binary classification** (informativeness of tweets). It also includes tools for **data processing**, **training**, **inference**, and **visualization** of results.

This project is only possible thanks to the [CrisisBench project](https://ojs.aaai.org/index.php/ICWSM/article/view/18115).

---

## Installation

1. Clone the repository:

```console
git clone https://github.com/kazgar/crisis_related_tweets_analysis
cd crisis_related_tweets_analysis
```

2. Install Python dependencies from `requirements.txt`:

```console
pip install -r requirements.txt
```

3. Install the `tweet_classification` package in **editable mode**:
```console
pip install -e .
```


> This allows any changes to the code in `tweet_classification/` to be reflected immediately without reinstalling.

---

## Data

This project uses **two datasets**:

1. **Multiclass tweet content classification**
   Categories include: `donation_and_volunteering`, `requests_or_needs`, `sympathy_and_support`, etc.

2. **Binary tweet informativeness classification**
   Labels: `informative` vs `not informative`.

### Getting the Data

<!-- Fill this section with instructions on how to obtain or download the datasets -->

### Processing the Data

To process raw datasets and prepare them for training, run the `notebooks/data_processing.ipynb` notebook.


---

## Training

Training scripts are located in `tweet_classification/`:

- **Multiclass humanitarian tweet classification**:
```console
python -m tweet_classification.human_train
```

- **Binary informativeness classification**:
```console
python -m tweet_classification.info_train
```


All training constants (`DROPOUT`, `SEED`, `EPOCH_NUM`, `BATCH_SIZE`, etc.) can be found in `tweet_classification/constants.py`.


> `HUMAN_EXPERIMENT_NR` and `INFO_EXPERIMENT_NR` are used to track consecutive experiments and automatically save results in corresponding directories.

---

## Visualizing Training Progress

To visualize training metrics such as loss and accuracy, run:
```console
notebooks/plot_results.ipynb
```


---

## Inference & Testing

Evaluate models on test data using:

- **Multiclass evaluation**:
```console
python -m tweet_classification.human_test
```
- **Binary evaluation**:
```console
python -m tweet_classification.info_test
```

---

## Visualizing Inference Performance

To visualize predictions and confusion matrices, run:
```console
notebooks/visualize_inference.ipynb
```

---

## Notes

- Make sure to set constants in `tweet_classification/constants.py` to control hyperparameters and experiment tracking.
- Editable installation (`pip install -e .`) allows changes to package code without reinstalling.
- This repository extends the methodology from the [CrisisBench project](https://ojs.aaai.org/index.php/ICWSM/article/view/18115).
