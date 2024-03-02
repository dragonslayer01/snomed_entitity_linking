from itertools import combinations

import dill as pickle
import evaluate
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import Dataset
from gensim.models.keyedvectors import KeyedVectors
from ipymarkup import show_span_line_markup
from more_itertools import chunked
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DebertaV2ForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

from snomed_graph import *
from train_cer import *

random_seed = 42  # For reproducibility
max_seq_len = 512  # Maximum sequence length for (BERT-based) encoders
cer_model_id = "microsoft/deberta-v3-large"  # Base model for Clinical Entity Recogniser
kb_embedding_model_id = ("sentence-transformers/all-MiniLM-L6-v2") # base model for concept encoder
use_LoRA = False  # Whether to use a LoRA to fine-tune the CER model

def train_kb_model():
  SG = SnomedGraph.from_serialized("../snomed_graph/full_concept_graph.gml")
  # If we want to load all of the concepts that were in scope of the annotation exercise, it's this:
  concepts_in_scope = (
      SG.get_descendants(71388002)
      | SG.get_descendants(123037004)
      | SG.get_descendants(404684003)
  )
  print(f"{len(concepts_in_scope)} concepts have been selected.")

  concepts_in_scope = [
    SG.get_concept_details(a) for a in annotations_df.concept_id.unique()
  ]

  kb_model = SentenceTransformer(kb_embedding_model_id)

  kb_sft_examples = [
      InputExample(texts=[syn1, syn2], label=1)
      for concept in tqdm(concepts_in_scope)
      for syn1, syn2 in combinations(concept.synonyms, 2)
  ]
  
  kb_sft_dataloader = DataLoader(kb_sft_examples, shuffle=True, batch_size=32)
  
  kb_sft_loss = losses.ContrastiveLoss(kb_model)
  
  kb_model.fit(
      train_objectives=[(kb_sft_dataloader, kb_sft_loss)],
      epochs=2,
      warmup_steps=100,
      checkpoint_path="~/temp/ke_encoder",
  )
  
  kb_model.save("kb_model")

  return kb_model

class Linker:
    def __init__(self, encoder, context_window_width=0):
        self.encoder = encoder
        self.entity_index = KeyedVectors(self.encoder[1].word_embedding_dimension)
        self.context_index = dict()
        self.history = dict()
        self.context_window_width = context_window_width

    def add_context(self, row):
        window_start = max(0, row.start - self.context_window_width)
        window_end = min(row.end + self.context_window_width, len(row.text))
        return row.text[window_start:window_end]

    def add_entity(self, row):
        return row.text[row.start : row.end]

    def fit(self, df=None, snomed_concepts=None):
        # Create a map from the entities to the concepts and contexts in which they appear
        if df is not None:
            for row in df.itertuples():
                entity = self.add_entity(row)
                context = self.add_context(row)
                map_ = self.history.get(entity, dict())
                contexts = map_.get(row.concept_id, list())
                contexts.append(context)
                map_[row.concept_id] = contexts
                self.history[entity] = map_

        # Add SNOMED CT codes for lookup
        if snomed_concepts is not None:
            for c in snomed_concepts:
                for syn in c.synonyms:
                    map_ = self.history.get(syn, dict())
                    contexts = map_.get(c.sctid, list())
                    contexts.append(syn)
                    map_[c.sctid] = contexts
                    self.history[syn] = map_

        # Create indexes to help disambiguate entities by their contexts
        for entity, map_ in tqdm(self.history.items()):
            keys = [
                (concept_id, occurance)
                for concept_id, contexts in map_.items()
                for occurance, context in enumerate(contexts)
            ]
            contexts = [context for contexts in map_.values() for context in contexts]
            vectors = self.encoder.encode(contexts)
            index = KeyedVectors(self.encoder[1].word_embedding_dimension)
            index.add_vectors(keys, vectors)
            self.context_index[entity] = index

        # Now create the top-level entity index
        keys = list(self.history.keys())
        vectors = self.encoder.encode(keys)
        self.entity_index.add_vectors(keys, vectors)

    def link(self, row):
        entity = self.add_entity(row)
        context = self.add_context(row)
        vec = self.encoder.encode(entity)
        nearest_entity = self.entity_index.most_similar(vec, topn=1)[0][0]
        index = self.context_index.get(nearest_entity, None)

        if index:
            vec = self.encoder.encode(context)
            key, score = index.most_similar(vec, topn=1)[0]
            sctid, _ = key
            return sctid
        else:
            return None


def evaluate_linker(linker, df):
    n_correct = 0
    n_examples = df.shape[0]

    for _, row in tqdm(df.iterrows(), total=n_examples):
        sctid = linker.link(row)
        if row["concept_id"] == sctid:
            n_correct += 1

    return n_correct / n_examples

def train_linker():
  kb_model = train_kb_model()
  linker_training_df = training_notes_df.join(training_annotations_df)
  linker_test_df = test_notes_df.join(test_annotations_df

  for context_window_width in tqdm([5, 8, 10, 12]):
    linker = Linker(kb_model, context_window_width)
    linker.fit(linker_training_df, concepts_in_scope)
    acc = evaluate_linker(linker, linker_test_df)
    print(f"Context Window Width: {context_window_width}\tAccuracy: {acc}")

  linker = Linker(kb_model, 12)
  linker.fit(linker_training_df, concepts_in_scope)
  
  with open("linker.pickle", "wb") as f:
      pickle.dump(linker, f)

