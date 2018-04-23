import numpy as np
import pandas as pd
import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import json
import sklearn
from sklearn import preprocessing as skpp

@registry.register_problem
class LyricClassification(text_problems.Text2ClassProblem):
    """Predict lyric genre from text"""

    @property
    def approx_vocab_size(self):
        return 30000

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def num_classes(self):
        return 11

    @property
    def class_labels(self, data_dir):
        del data_dir
        data = pd.read_csv("./dataset/genre_id_lyrics.csv")
        mappings = data[['genre', 'genre_id']].drop_duplicates()
        map_list = sorted([(genre_id, genre) for genre, genre_id in mappings.values])
        return [genre for genre_id, genre in map_list]
    
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        
        data = pd.read_csv('./dataset/genre_id_lyrics.csv')
        lyrics_to_id = data[['lyrics', 'genre_id']].values
        for lyric, genre_id in lyrics_to_id:
            yield {
                "inputs": lyric,
                "label": genre_id
            }