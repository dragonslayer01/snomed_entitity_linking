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

# Creating Training and Testing Dataset

# Step through the annotation spans for a given note.  When they're exhausted,
# return (1000000, 1000000).  This will avoid a StopIteration exception.

def get_annotation_boundaries(note_id, annotations_df):
    for row in annotations_df.loc[note_id].itertuples():
        yield row.start, row.end, row.concept_id
    yield 1000000, 1000000, None

def generate_ner_dataset(notes_df, annotations_df):
    for row in notes_df.itertuples():
        tokenized = cer_tokenizer(
            row.text,
            return_offsets_mapping=False,  # Avoid misalignments due to destructive tokenization
            return_token_type_ids=False,  # We're going to construct these below
            return_attention_mask=False,  # We'll construct this by hand
            add_special_tokens=False,  # We'll add these by hand
            truncation=False,  # We'll chunk the notes ourselves
        )

        # Prime the annotation generator and fetch the token <-> word_id map
        annotation_boundaries = get_annotation_boundaries(row.Index, annotations_df)
        ann_start, ann_end, concept_id = next(annotation_boundaries)
        word_ids = tokenized.word_ids()

        # The offsets_mapping returned by the tokenizer will be misaligned vs the original text.
        # This is due to the fact that the tokenization scheme is destructive, for example it
        # drops spaces which cannot be recovered when decoding the inputs.
        # In the following code snippet we create an offset mapping which is aligned with the
        # original text; hence we can accurately locate the annotations and match them to the
        # tokens.
        global_offset = 0
        global_offset_mapping = []

        for input_id in tokenized["input_ids"]:
            token = cer_tokenizer.decode(input_id)
            pos = row.text[global_offset:].find(token)
            start = global_offset + pos
            end = global_offset + pos + len(token)
            global_offset = end
            global_offset_mapping.append((start, end))

        # Note the max_seq_len - 2.
        # This is because we will have to add [CLS] and [SEP] tokens once we're done.
        it = zip(
            chunked(tokenized["input_ids"], max_seq_len - 2),
            chunked(global_offset_mapping, max_seq_len - 2),
            chunked(word_ids, max_seq_len - 2),
        )

        # Since we are chunking the discharge notes, we need to maintain the start and
        # end character index for each chunk so that we can align the annotations for
        # chunks > 1
        chunk_start_idx = 0
        chunk_end_idx = 0

        for chunk_id, chunk in enumerate(it):
            input_id_chunk, offset_mapping_chunk, word_id_chunk = chunk
            token_type_chunk = list()
            concept_id_chunk = list()
            prev_word_id = -1
            concept_word_number = 0
            chunk_start_idx = chunk_end_idx
            chunk_end_idx = offset_mapping_chunk[-1][1]

            for offsets, word_id in zip(offset_mapping_chunk, word_id_chunk):
                token_start, token_end = offsets

                # Check whether we need to fetch the next annotation
                if token_start >= ann_end:
                    ann_start, ann_end, concept_id = next(annotation_boundaries)
                    concept_word_number = 0

                # Check whether the token's position overlaps with the next annotation
                if token_start < ann_end and token_end > ann_start:
                    if prev_word_id != word_id:
                        concept_word_number += 1

                    # If so, annotate based on the word number in the concept
                    if concept_word_number == 1:
                        token_type_chunk.append(label2id["B-clinical_entity"])
                    else:
                        token_type_chunk.append(label2id["I-clinical_entity"])

                    # Add the SCTID (we'll use this later to train the Linker)
                    concept_id_chunk.append(concept_id)

                # Not part of an annotation
                else:
                    token_type_chunk.append(label2id["O"])
                    concept_id_chunk.append(None)

                prev_word_id = word_id

            # Manually adding the [CLS] and [SEP] tokens.
            token_type_chunk = [-100] + token_type_chunk + [-100]
            input_id_chunk = (
                [cer_tokenizer.cls_token_id]
                + input_id_chunk
                + [cer_tokenizer.sep_token_id]
            )
            attention_mask_chunk = [1] * len(input_id_chunk)
            offset_mapping_chunk = (
                [(None, None)] + offset_mapping_chunk + [(None, None)]
            )
            concept_id_chunk = [None] + concept_id_chunk + [None]

            yield {
                # These are the fields we need
                "note_id": row.Index,
                "input_ids": input_id_chunk,
                "attention_mask": attention_mask_chunk,
                "labels": token_type_chunk,
                # These fields are helpful for debugging
                "chunk_id": chunk_id,
                "chunk_span": (chunk_start_idx, chunk_end_idx),
                "offset_mapping": offset_mapping_chunk,
                "text": row.text[chunk_start_idx:chunk_end_idx],
                "concept_ids": concept_id_chunk,
            }


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_cer_model():
  """
  Trigger Function to finetune CER model
  """

  
    random_seed = 42  # For reproducibility
    max_seq_len = 512  # Maximum sequence length for (BERT-based) encoders
    cer_model_id = "microsoft/deberta-v3-large"  # Base model for Clinical Entity Recogniser
    kb_embedding_model_id = ("sentence-transformers/all-MiniLM-L6-v2") # base model for concept encoder
    use_LoRA = False  # Whether to use a LoRA to fine-tune the CER model
    
    
    torch.manual_seed(random_seed)
    assert torch.cuda.is_available()
    
    
    notes_df = pd.read_csv("data/training_notes.csv").set_index("note_id")
    print(f"{notes_df.shape[0]} notes loaded.")
    
    
    annotations_df = pd.read_csv("data/training_annotations.csv").set_index("note_id")
    print(f"{annotations_df.shape[0]} annotations loaded.")
    print(f"{annotations_df.concept_id.nunique()} unique concepts seen.")
    print(f"{annotations_df.index.nunique()} unique notes seen.")
    
    
    training_notes_df, test_notes_df = train_test_split(
        notes_df, test_size=32, random_state=random_seed
    )
    training_annotations_df = annotations_df.loc[training_notes_df.index]
    test_annotations_df = annotations_df.loc[test_notes_df.index]
    
    print(
        f"There are {training_annotations_df.shape[0]} total annotations in the training set."
    )
    print(f"There are {test_annotations_df.shape[0]} total annotations in the test set.")
    print(
        f"There are {training_annotations_df.concept_id.nunique()} distinct concepts in the training set."
    )
    print(
        f"There are {test_annotations_df.concept_id.nunique()} distinct concepts in the test set."
    )
    print(f"There are {training_notes_df.shape[0]} notes in the training set.")
    print(f"There are {test_notes_df.shape[0]} notes in the test set.")
    
    
    label2id = {"O": 0, "B-clinical_entity": 1, "I-clinical_entity": 2}
    
    id2label = {v: k for k, v in label2id.items()}
    
    cer_tokenizer = AutoTokenizer.from_pretrained(
        cer_model_id, model_max_length=max_seq_len
    )
    
    # We can ignore the "Token indices sequence length is longer than the specified maximum sequence length"
    # warning because we are chunking by hand.
    train = pd.DataFrame(
        list(generate_ner_dataset(training_notes_df, training_annotations_df))
    )
    train = Dataset.from_pandas(train)
    
    test = pd.DataFrame(list(generate_ner_dataset(test_notes_df, test_annotations_df)))
    test = Dataset.from_pandas(test)
    
    # The data collator handles batching for us.
    data_collator = DataCollatorForTokenClassification(tokenizer=cer_tokenizer)
    
    seqeval = evaluate.load("seqeval")
    
    cer_model = DebertaV2ForTokenClassification.from_pretrained(
        cer_model_id, num_labels=3, id2label=id2label, label2id=label2id
    )
    
    if use_LoRA:
        lora_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="TOKEN_CLS",
        )
    
        cer_model = get_peft_model(cer_model, lora_config)
    
        cer_model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="~/temp/cer_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=False,
        seed=random_seed,
    )
    
    trainer = Trainer(
        model=cer_model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=cer_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    trainer.save_model("cer_model")
    cer_tokenizer.save_pretrained("cer_model")

def infer_cer(use_LoRA = True,model_path = "cer_model"):
    """
    Infer Pipeline for trained CER Model
    """
    if use_LoRA:
        config = PeftConfig.from_pretrained(model_path)
    
        cer_model = DebertaV2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=config.base_model_name_or_path,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        )
        cer_model = PeftModel.from_pretrained(cer_model, "cer_model")
    else:
        cer_model = DebertaV2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        )

    cer_pipeline = pipeline(
    task="token-classification",
    model=cer_model,
    tokenizer=cer_tokenizer,
    aggregation_strategy="first",
    device="cpu",
    )

    return cer_pipeline



