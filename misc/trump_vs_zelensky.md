
# Prepare the data


```python
import os
import re
import pandas as pd 
import numpy as np
import requests

import tabled
```


```python
# settings
raw_src_url = 'https://raw.githubusercontent.com/thorwhalen/content/refs/heads/master/text/trump-zelensky-2025-03-01--with_speakers.txt'

rootdir = '.'   # NOTE: Put your own rootdir here

# save keys (e.g. relative paths)
embeddings_save_key = 'data/trump_vs_zelenskyy_embeddings.parquet'
transcript_save_key = 'data/trump_vs_zelenskyy_transcript.parquet'
```


```python
store = tabled.DfFiles(rootdir)
```


```python
# Get the text of the transcript
transcript_text = requests.get(raw_src_url).text
```


```python
# Every line of transcript_text starts with [speaker]: [text]
# Let's parse the transcript_text to get a list of (speaker, text) dicts
# Define a regex pattern to match the speaker and text
pattern = re.compile(r'\[(?P<speaker>[^\]]+)\]: (?P<text>.*)')

# Parse the transcript_text to get a list of (speaker, text) dicts
transcript_dict_list = [
    match.groupdict()
    for match in pattern.finditer(transcript_text)
]

transcript_df = pd.DataFrame(transcript_dict_list)
transcript_df
```






```python
t = transcript_df['speaker'].value_counts()
n_top_speakers = 4
print(f"Unique speakers: {len(t)}")
print(f"Top 5 speakers: {t.head(n_top_speakers)}")
```

    Unique speakers: 15
    Top 5 speakers: speaker
    Trump         509
    Zelenskyy     251
    Vance          32
    SPEAKER_17      9
    Name: count, dtype: int64



```python
# replace all speakers that are not the top 3 with 'Other'
top_speakers = transcript_df['speaker'].value_counts().head(3).index
transcript_df['speaker'] = transcript_df['speaker'].apply(
    lambda speaker: speaker if speaker in top_speakers else 'Other'
)
transcript_df['speaker'].value_counts()  # only 4 unique speakers now
```




    speaker
    Trump        509
    Zelenskyy    251
    Other         66
    Vance         32
    Name: count, dtype: int64




```python
if embeddings_save_key not in store:
    # compute the embeddings
    import oa
    embeddings_vectors = oa.embeddings(transcript_df['text'])
    embeddings_df = pd.DataFrame({'embeddings': embeddings_vectors})
    store[embeddings_save_key] = embeddings_df
else:
    embeddings_df = store[embeddings_save_key]
    embeddings_vectors = np.vstack(embeddings_df['embeddings'])
```


```python
# project embeddings to plane using TSNE
if 'tsne_x' not in transcript_df.columns:

    from sklearn.manifold import TSNE
    
    tsne_vectors = TSNE(n_components=2).fit_transform(embeddings_vectors)

    t = pd.DataFrame(tsne_vectors, columns=['tsne_x', 'tsne_y'])
    transcript_df = pd.concat([transcript_df, t], axis=1)

print(f"{transcript_df.shape=}")
transcript_df.iloc[0]
```

    transcript_df.shape=(858, 4)





    speaker                         Trump
    text       Well, thank you very much.
    tsne_x                     -16.044361
    tsne_y                      33.261379
    Name: 0, dtype: object




```python
# project embeddings to plane using linear discriminant analysis on speakers
if 'lda_x' not in transcript_df.columns:

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA

    speakers = transcript_df['speaker']

    pipeline = Pipeline([
        ('pca', PCA(n_components=50)),
        ('lda', LDA(n_components=2))
    ])

    pipeline.fit(embeddings_vectors, y=speakers)
    lda_vectors = pipeline.transform(embeddings_vectors)

    t = pd.DataFrame(lda_vectors, columns=['lda_x', 'lda_y'])
    transcript_df = pd.concat([transcript_df, t], axis=1)

print(f"{transcript_df.shape=}")
transcript_df.iloc[0]
```

    transcript_df.shape=(858, 6)





    speaker                         Trump
    text       Well, thank you very much.
    tsne_x                     -16.044361
    tsne_y                      33.261379
    lda_x                        0.788964
    lda_y                       -0.248835
    Name: 0, dtype: object




```python
# project embeddings to plane using linear discriminant analysis on speakers
if 'single_speaker_lda' not in transcript_df.columns:

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA

    speakers = transcript_df['speaker']

    pipeline = Pipeline([
        ('pca', PCA(n_components=50)),
        ('lda', LDA(n_components=1))
    ])

    pipeline.fit(embeddings_vectors, y=speakers)
    lda_vectors = pipeline.transform(embeddings_vectors)

    t = pd.DataFrame(lda_vectors, columns=['single_speaker_lda'])
    transcript_df = pd.concat([transcript_df, t], axis=1)

print(f"{transcript_df.shape=}")
transcript_df.iloc[0]
```

    transcript_df.shape=(858, 19)





    speaker                                    Trump
    text                  Well, thank you very much.
    tsne_x                                -16.044361
    tsne_y                                 33.261379
    lda_x                                   0.788964
    lda_y                                  -0.248835
    pca_1                                  -0.206804
    pca_2                                   0.045916
    neg                                          0.0
    neu                                        0.395
    pos                                        0.605
    compound                                  0.5574
    Happy                                        0.0
    Angry                                        0.0
    Surprise                                     0.0
    Sad                                          0.0
    Fear                                         0.0
    turn                                           0
    single_speaker_lda                      0.849894
    Name: 0, dtype: object




```python
if 'pca_1' not in transcript_df.columns:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    pca_vectors = pca.fit_transform(embeddings_vectors)

    t = pd.DataFrame(pca_vectors, columns=['pca_1', 'pca_2'])
    transcript_df = pd.concat([transcript_df, t], axis=1)
```


```python
ww = list(map(sentiment_score, transcript_df.iloc[:3]['text']))
ww
```




    [0.9594653844833374, 0.9941986799240112, 0.998121440410614]




```python
if 'sentiment_f' not in transcript_df.columns:
    from mood.sentiment import flair_sentiment_score

    t = pd.DataFrame(list(map(flair_sentiment_score, transcript_df['text'])), columns=['sentiment_f'])
    transcript_df = pd.concat([transcript_df, t], axis=1)
transcript_df.iloc[0]
```




    speaker                             Trump
    text           Well, thank you very much.
    tsne_x                         -16.044361
    tsne_y                          33.261379
    lda_x                            0.788964
    lda_y                           -0.248835
    pca_1                           -0.206804
    pca_2                            0.045916
    neg                                   0.0
    neu                                 0.395
    pos                                 0.605
    compound                           0.5574
    Happy                                 0.0
    Angry                                 0.0
    Surprise                              0.0
    Sad                                   0.0
    Fear                                  0.0
    turn                                    0
    sentiment_f                      0.959465
    Name: 0, dtype: object




```python
import dol

pickle_store = dol.PickleFiles('.')
models = pickle_store['oa_embeddings_sentiment_models.pickle']

print(f"{list(models)=}")

label = 'anger'
print(f"{list(models[label])=}")
print(f"{models[label]['stats']}")

if 'disgust' not in transcript_df.columns:
    for sentiment, d in models.items():
        model = d['model']
        model.predict()

```

    list(models)=['anger', 'sadness', 'surprise', 'disgust', 'fear']
    list(models[label])=['label', 'model', 'stats']
    
    === Model Performance Statistics ===
    Accuracy:  0.8103
    Precision: 0.6691
    Recall:    0.6964
    F1 Score:  0.6825
    ROC AUC:   0.8603
    
    Confusion Matrix:
    [[812 135]
     [119 273]]
    
    Detailed Classification Report:
                  precision    recall  f1-score   support
    
             0.0       0.87      0.86      0.86       947
             1.0       0.67      0.70      0.68       392
    
        accuracy                           0.81      1339
       macro avg       0.77      0.78      0.77      1339
    weighted avg       0.81      0.81      0.81      1339
    
    
    Test Size: 0.2, Random State: 42
    Feature Scaling: Applied
    Training Time: 2.2221 seconds
    



```python
if 'compound' not in transcript_df.columns:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    t = pd.DataFrame(list(map(analyzer.polarity_scores, transcript_df['text'])))
    transcript_df = pd.concat([transcript_df, t], axis=1)

print(f"{transcript_df.shape=}")
transcript_df.iloc[0]
```

    transcript_df.shape=(858, 12)





    speaker                          Trump
    text        Well, thank you very much.
    tsne_x                      -16.044361
    tsne_y                       33.261379
    lda_x                         0.788964
    lda_y                        -0.248835
    pca_1                        -0.206804
    pca_2                         0.045916
    neg                                0.0
    neu                              0.395
    pos                              0.605
    compound                        0.5574
    Name: 0, dtype: object




```python
if 'Happy' not in transcript_df.columns:
    import text2emotion as te

    t = pd.DataFrame(list(map(te.get_emotion, transcript_df['text'])))
    transcript_df = pd.concat([transcript_df, t], axis=1)

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/thorwhalen/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/thorwhalen/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/thorwhalen/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
transcript_df['turn'] = range(len(transcript_df))
```


```python
print(f"{transcript_df.shape=}")
transcript_df.iloc[0]
```

    transcript_df.shape=(858, 18)





    speaker                          Trump
    text        Well, thank you very much.
    tsne_x                      -16.044361
    tsne_y                       33.261379
    lda_x                         0.788964
    lda_y                        -0.248835
    pca_1                        -0.206804
    pca_2                         0.045916
    neg                                0.0
    neu                              0.395
    pos                              0.605
    compound                        0.5574
    Happy                              0.0
    Angry                              0.0
    Surprise                           0.0
    Sad                                0.0
    Fear                               0.0
    turn                                 0
    Name: 0, dtype: object




```python
store[transcript_save_key] = transcript_df
```
