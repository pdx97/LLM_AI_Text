#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd
import gc

# Load data
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

# Data preprocessing
excluded_prompt_name_list = ['Distance learning', 'Grades for extracurricular activities', 'Summer projects']
train = train[~train['prompt_name'].isin(excluded_prompt_name_list)].drop_duplicates(subset=['text']).reset_index(drop=True)

# BPE Tokenizer
LOWERCASE = False
VOCAB_SIZE = 14_000_000
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens
)
dataset = Dataset.from_pandas(test[['text']])

# Using an iterator for training
train_iterator = (text for text in dataset['text'])
raw_tokenizer.train_from_iterator(train_iterator, trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Tokenize test set with new tokenizer
tokenized_texts_test = [tokenizer.tokenize(text) for text in tqdm(test['text'].tolist())]

# Tokenize train set
tokenized_texts_train = [tokenizer.tokenize(text) for text in tqdm(train['text'].tolist())]

# Fitting TfidfVectorizer on the test set
vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    analyzer='word',
    tokenizer=lambda x: x,  # No need for a dummy function
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents='unicode'
)
vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

# Use the vocabulary from the test set to fit the vectorizer on the train set
vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    vocabulary=vocab,
    analyzer='word',
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents='unicode'
)
tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)
del vectorizer
gc.collect()

y_train_label = train['label'].values

if len(test.text.values) <= 5:
    sub.to_csv('submission.csv', index=False)
else:
    # Classifiers
    clf = MultinomialNB(alpha=0.0225)
    sgd_model = SGDClassifier(
        max_iter=9000,
        tol=1e-4,
        random_state=6743,
        loss="modified_huber"
    )
    p = {
        'verbose': -1,
        'n_iter': 3000,
        'colsample_bytree': 0.7800,
        'colsample_bynode': 0.8000,
        'random_state': 6743,
        'metric': 'auc',
        'objective': 'cross_entropy',
        'learning_rate': 0.00581909898961407,
    }
    lgb = LGBMClassifier(**p)
    cat = CatBoostClassifier(
        iterations=3000,
        verbose=0,
        subsample=0.35,
        random_seed=6543,
        allow_const_label=True,
        loss_function='CrossEntropy',
        learning_rate=0.005599066836106983,
    )

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('mnb', clf), ('sgd', sgd_model), ('lgb', lgb), ('cat', cat)],
        weights=[0.1, 0.31, 0.28, 0.67],
        voting='soft',
        n_jobs=-1
    )

    ensemble.fit(tf_train, y_train_label)
    gc.collect()

    # Predictions
    final_preds = ensemble.predict_proba(tf_test)[:, 1]
    sub['generated'] = final_preds
    sub.to_csv('submission.csv', index=False)
    sub.head()


# In[2]:


print("DONE")


# In[ ]:




