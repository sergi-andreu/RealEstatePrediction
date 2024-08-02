# Introduction

This is the code for an assignment on predicting the price of buildings from ads of a real estate platform.
The code found here is far for productive, and is designed to follow my though process during this task.

the main codes are to be found in the notebooks folder, and they are order by the thought process that I followed.


# Evaluation recommendation

For a thourough evaluation, going through all the notebooks, in order, would be recommended. I tried to add everything contributing to understand my thought process.
Time is valuable, so looking through it is also welcome.

My recommendation would be to look at:
- *src/data_preprocessing*
- *12_Final_model_evaluation.ipynb*


## Repository structure

```plaintext
│
├── data/
│
├── notebooks/
│ ├── catboost_info/
│ ├── wandb/
│ ├── 00_First_data_impression.ipynb
│ ├── 01_Extract_features_from_params.ipynb
│ ├── 02_00_Extract_features_from_text_Word2Vec.ipynb
│ ├── 02_01_Extract_features_from_text_BERT.ipynb
│ ├── 02_02_Reduce_embedding_dimensionality.ipynb
│ ├── 03_Extract_time_features.ipynb
│ ├── 04_Extract_map_features.ipynb
│ ├── 05_Load_and_extract_data_using_scripts.ipynb
│ ├── 06_Data_exploration.ipynb
│ ├── 07_Baseline_models_training_with_outliers.ipynb
│ ├── 07_Baseline_models_training_without_outliers.ipynb
│ ├── 08_Catboost_models_training_with_outliers.ipynb
│ ├── 09_Catboost_hyperparameter_tuning.ipynb
│ ├── 10_Catboost_feature_elimination.ipynb
│ ├── 11_Final_model_training.ipynb
│ ├── 12_Final_model_evaluation.ipynb
│ └── 13_Final_model_inference.ipynb
│
├── src/
│ ├── BERT_embeddings.py
│ ├── compute_metrics.py
│ ├── data_preprocessing.py
│ ├── outlier_imputer.py
│ ├── params_parser.py
│ └── word2vec_embeddings.py
│
├── .gitignore
├── readme.md
├── requirements_freeze.txt
└── requirements.txt
```


# Getting started

## Prepare the dependencies

You can create a python environment and install dependencies by:

```
pip install -r requirements.txt
```
And execute the notebooks (as you usually do).

The versions are not specified (for time reasons; sorry). But, to make sure for reproducibility, I have pip-freezed the dependencies in the *requirements.txt_freeze* file.

So, if it is not working, do
```
pip install -r requirements_freeze.txt
```

## Get the data
To obtain lots of required files from the **data/** folder, one would need to re-run lots of notebooks, which is not quite nice.

For that reason, please ask me for a *.zip* of my pre-computed data folder.
I have not uploaded the content here, since I don't want to leak data (and also I don't like to upload binaries / large files to github).


## Use the final model

Using the final model for inference is not recommended, mostly for efficiency reasons.<br>

However, if you really want to do this, you would have to use the [*13_Final_model_inference* notebook](notebooks/13_Final_model_inference.ipynb). 


# Future work

### Meta-model training
Solving these bussiness challenge is not just training the model, but understanding the overall product.
In this sense, there are bussiness (product) decisions that would affect the behaviour of the model, and therefore we should change some training or hyperparameter decisions.

Some questions to ask are:
- ***Should we recommend 'standard' prices, or succesful prices?***: By filtering out data of properties that have not been sold, and also spam / other ads, we could give a better recommendation of the price to stipulate in the ad. Moreover, we could even make more advanced models, trying to predict when the property would be sold, for different prices.
- ***When would the customer get a recommendation on the price?***: If the recommendation is given in batch (whenever they click some button) is not the same as if we give some alerts whenever a posting of theirs is not getting attention / needs to be updated. This decision could change temporal features (created_at_first, updated_at)
- ***When would this model be re-trained, and with which periodicity***: this could affect the decision on which temporal features to use (features gotten from created_at_first, updated_at). For example, if training quite often, we could standardize the updated_at feature often. But, if not, we would be inferring data for an out-of-distribution feature.
- ***Do we want an explainable model?***: If we want to be 

### Feature transformation
- When extracting features from the title and description, we removed numeric characters. The presence of these can indeed change the value of the properties. Studying the behaviour with respect to the presence of numbers, and generating features from this, could be interesting.
- One could use pre-trained Word2Vec models for extracting features from title and description, such as the ones in: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/. However, for time constraints, finding a good, lightweight model is not trivial.