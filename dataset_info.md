```python
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("dair-ai/emotion")

print(ds_builder.info.description)

print(ds_builder.info.features)
```


 # rotten_tomatoes
  Movie Review Dataset.
  This is a dataset of containing 5,331 positive and 5,331 negative processed
  sentences from Rotten Tomatoes movie reviews. This data was first used in Bo
 Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for
  sentiment categorization with respect to rating scales.'', Proceedings of the
  ACL, 2005.
 {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}

 # ag_news
  AG is a collection of more than 1 million news articles. News articles have been
  gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
  activity. ComeToMyHead is an academic news search engine which has been running
  since July, 2004. The dataset is provided by the academic comunity for research
  purposes in data mining (clustering, classification, etc), information retrieval
  (ranking, search, etc), xml, data compression, data streaming, and any other
  non-commercial activity. For more information, please refer to the link
  http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .
  The AG's news topic classification dataset is constructed by Xiang Zhang
  (xiang.zhang@nyu.edu) from the dataset above. It is used as a text
  classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
  LeCun. Character-level Convolutional Networks for Text Classification. Advances
  in Neural Information Processing Systems 28 (NIPS 2015).
  {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech'], id=None)}

  # go_emotions
  The GoEmotions dataset contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.
  The emotion categories are admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
  disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness,
  optimism, pride, realization, relief, remorse, sadness, surprise.
  {'text': Value(dtype='string', id=None), 'labels': Sequence(feature=ClassLabel(names=['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'], id=None), length=-1, id=None), 'id': Value(dtype='string', id=None)}

  # dair-ai/emotion
  Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. For more detailed information please refer to the paper.
  {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}
