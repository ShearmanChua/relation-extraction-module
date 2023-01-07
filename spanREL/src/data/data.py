import json
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Callable, List, Set, Tuple, Dict, Any
import ipdb
from nltk.tokenize import word_tokenize
from collections import OrderedDict

def add_marker_tokens(tokenizer, ner_labels, tokenizer_path):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>','<ENT_START>','<ENT_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
        new_tokens.append('<ENT_START=%s>'%label)
        new_tokens.append('<ENT_END=%s>'%label)

    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
        new_tokens.append('<ENT=%s>'%label)
    # print("\n")
    # print("Entity tokens: ",new_tokens)
    tokenizer.add_tokens(new_tokens)
    tokenizer.save_pretrained(tokenizer_path)

    return tokenizer


def enumerate_spans(
    sentence: List,
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.
    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.
    # Parameters
    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end, (end-start)))
    return spans


class RelationDataset(Dataset):
    def __init__(self, cfg: Any, json_file: str, tokenizer: Any, relation_labels: List, entity_labels: List, split: str):
        self.cfg = cfg
        self.split = split
        
        if cfg.add_ner_tokens and split=='train': # 
            print("\n")
            print("Adding entity markers to tokenizer!!!")
            tokenizer = add_marker_tokens(tokenizer,entity_labels, cfg.longformer.autotokenizer)

        if os.path.exists(os.path.join(cfg.output_dir, 'special_tokens.json')):
            with open(os.path.join(cfg.output_dir, 'special_tokens.json'), 'r') as f:
                self.special_tokens = json.load(f)
        else:
            self.special_tokens = {}

        self.tokenizer = tokenizer
        self.relation_labels = relation_labels
        self.consolidated_dataset, self.global_labels = self._read(json_file)

    def _read(self, json_file: str) -> Tuple[List[Dict], List]:
        if self.cfg.debug:
            gold_docs = [json.loads(line) for idx, line in enumerate(
                open(json_file)) if idx < 50]
        else:
            gold_docs = [json.loads(line) for line in open(json_file)]
        encoded_gold_docs = self.encode(gold_docs)
        encoded_gold_docs_w_spanpairs_labels, global_labels = self.get_entity_pairs(
            encoded_gold_docs)
        return encoded_gold_docs_w_spanpairs_labels, global_labels

    def get_entity_pairs(self, docs: List[Dict]) -> Tuple[List[Dict], List]:
        docs_w_span_labels = []
        global_labels = []
        for doc in docs:
            labels = []
            entity_pairs = []

            positive_pairs = [pairs[:-1]
                              for pairs in doc['relations']]

            positive_labels = [self.relation_labels.index(pairs[-1])
                               for pairs in doc['relations']]

            if self.cfg.use_predicted_entities:
                entities = [ent_span for ent_span in doc['predicted_ner']
                            if ent_span[1] < self.cfg.max_length]
            else:
                entities = [ent_span for ent_span in doc['ner']
                            if ent_span[1] < self.cfg.max_length]

            # Iterate over all entity pairs
            for i in range(len(entities)):
                for j in range(i):
                    candidate_pair = entities[i][:2]+entities[j][:2]
                    candidate_pair_w_width = candidate_pair[:2]+[
                        candidate_pair[1]-candidate_pair[0]+1]+candidate_pair[2:]+[
                        candidate_pair[3]-candidate_pair[2]+1]                    

                    if candidate_pair in positive_pairs and candidate_pair_w_width[2] < self.cfg.max_span_length and candidate_pair_w_width[5] < self.cfg.max_span_length:
                        # add candidate
                        entity_pairs.append(candidate_pair_w_width)
                        # add label
                        label_idx = positive_pairs.index(candidate_pair)
                        labels.append(positive_labels[label_idx])
                    elif random.random() < self.cfg.negative_sample_ratio and candidate_pair_w_width[2] < self.cfg.max_span_length and candidate_pair_w_width[5] < self.cfg.max_span_length:
                        # add negative candidate
                        entity_pairs.append(candidate_pair_w_width)
                        # add negative label
                        labels.append(
                            self.relation_labels.index("NonRelation"))

            global_labels += labels
            docs_w_span_labels.append(
                {**doc, "span_pairs": entity_pairs, "labels": labels})

        return docs_w_span_labels, global_labels

    def encode(self, docs: List[Dict]) -> List[Dict]:

        special_tokens = {}

        def get_special_token(w, special_tokens, unused_tokens=False):
            if w not in special_tokens:
                if unused_tokens:
                    special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
                else:
                    special_tokens[w] = ('<' + w + '>')
            return special_tokens[w] 

        encoded_docs = []
        for doc in docs:
            if self.cfg.add_ner_tokens:
                tokens = doc["tokens"][0]
                ner = doc['predicted_ner'] if self.cfg.use_predicted_entities else doc["ner"]
                relations = doc["relations"]

                ner_ordered_dict = OrderedDict()
                for entity in ner:
                    ner_ordered_dict[int(str(entity[0])+str(entity[1]))] = entity

                ner_ordered_dict = OrderedDict(sorted(ner_ordered_dict.items()))

                ner_ordered = [value for key,value in ner_ordered_dict.items()]

                ner_start_type_dict = OrderedDict()
                ner_end_type_dict = OrderedDict()
                ner_start = []
                ner_end = []
                
                for entity in ner_ordered:
                    
                    if entity[0] in ner_start_type_dict.keys():
                        old_list = ner_start_type_dict[entity[0]].copy()
                        old_list.append(entity[2])
                        ner_start_type_dict.update({entity[0]: old_list})
                    else:
                        ner_start_type_dict[entity[0]] = [entity[2]]

                    ner_start.append(entity[0])

                    if entity[1] in ner_end_type_dict.keys():
                        old_list = ner_end_type_dict[entity[1]].copy()
                        old_list.append(entity[2])
                        ner_end_type_dict.update({entity[1]: old_list})
                    else:
                        ner_end_type_dict[entity[1]] = [entity[2]]

                    ner_end.append(entity[1])

                ner_start_type_dict = OrderedDict(sorted(ner_start_type_dict.items()))
                ner_end_type_dict = OrderedDict(sorted(ner_end_type_dict.items()))
                
                tokens_w_types = []
                new_start_map = OrderedDict()
                new_end_map = OrderedDict()

                for idx in range(0,len(tokens)+1):
                    if idx in ner_start:
                        count_tokens_added = 0
                        for entity_type in ner_start_type_dict[idx]:
                            tokens_w_types.append(get_special_token("ENT_START=%s" % entity_type,{}))
                            count_tokens_added += 1
                        tokens_w_types.append(tokens[idx])
                        if idx in ner_end:
                            for entity_type in ner_end_type_dict[idx]:
                                tokens_w_types.append(get_special_token("ENT_END=%s" % entity_type,{}))
                                count_tokens_added += 1
                            new_end_map[idx] = len(tokens_w_types)
                        new_start_map[idx] = len(tokens_w_types)-1-count_tokens_added
                    elif idx in ner_end:
                        for entity_type in ner_end_type_dict[idx]:
                            tokens_w_types.append(get_special_token("ENT_END=%s" % entity_type,{}))
                        if idx < len(tokens):
                            tokens_w_types.append(tokens[idx])
                            new_end_map[idx] = len(tokens_w_types)-1
                        else:
                            new_end_map[idx] = len(tokens_w_types)
                    elif idx < len(tokens):
                        tokens_w_types.append(tokens[idx])
                
                new_token_pos = []
                for ner in ner_ordered:
                    new_token_pos.append([new_start_map[ner[0]],new_end_map[ner[1]],ner[2]])

                new_relation_positions = []
                for relation in relations:
                    new_relation_positions.append([new_start_map[relation[0]],new_end_map[relation[1]],new_start_map[relation[2]],new_end_map[relation[3]],relation[4]])

                if self.cfg.use_predicted_entities:
                    doc.update({"predicted_ner": new_token_pos})
                else:
                    doc.update({"ner": new_token_pos})

                doc.update({"relations": new_relation_positions})

                text = self.tokenizer.convert_tokens_to_string(tokens_w_types)
            else:
                text = self.tokenizer.convert_tokens_to_string(doc["tokens"][0])

            encodings = self.tokenizer(
                text, padding="max_length", truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
            encoded_docs.append(
                {**doc, "input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]})
        return encoded_docs

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze(1)

        span_subj = [[span[:3] for span in ex['span_pairs']] for ex in batch]
        span_obj = [[span[3:] for span in ex['span_pairs']] for ex in batch]

        labels = [ex['labels'] for ex in batch]
        labels = torch.tensor(
            [label for sample in labels for label in sample])
        span_mask = torch.tensor([idx for idx, ex in enumerate(
            batch) for span_pairs in ex['span_pairs']])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_subj": span_subj,
            "span_obj": span_obj,
            "labels": labels,
            "span_mask": span_mask
        }

class InferenceDataset(Dataset):
    def __init__(self, cfg: Any, docs: List[Dict], tokenizer: Any, relation_labels: List, entity_labels: List):
        self.cfg = cfg

        if os.path.exists(os.path.join(cfg.output_dir, 'special_tokens.json')):
            with open(os.path.join(cfg.output_dir, 'special_tokens.json'), 'r') as f:
                self.special_tokens = json.load(f)
        else:
            self.special_tokens = {}

        self.tokenizer = tokenizer
        self.relation_labels = relation_labels
        self.consolidated_dataset, self.global_labels = self._read(docs)

    def _read(self, docs: List[Dict]) -> Tuple[List[Dict], List]:
        if self.cfg.debug:
            docs = [doc for idx, doc in enumerate(docs) if idx < 50]

        encoded_gold_docs = self.encode(docs)
        encoded_gold_docs_w_spanpairs_labels, global_labels = self.get_entity_pairs(
            encoded_gold_docs)
        return encoded_gold_docs_w_spanpairs_labels, global_labels

    def get_entity_pairs(self, docs: List[Dict]) -> Tuple[List[Dict], List]:
        def find_sub_list(sl,l):
            results=[]
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    results.append([ind,ind+sll])

            return results

        docs_w_span_labels = []
        global_labels = []
        for doc in docs:
            labels = []
            entity_pairs = []

            if 'relations' in doc.keys():

                positive_pairs = [pairs[:-1]
                                for pairs in doc['relations']]

                positive_labels = [self.relation_labels.index(pairs[-1])
                                for pairs in doc['relations']]

            else:
                positive_pairs = []

                positive_labels = []

            if self.cfg.use_predicted_entities:
                tokenized_spans = []
                exist_span = {}
                for ent_span in doc['predicted_ner']:
                    entity_tokens = self.tokenizer.tokenize(doc['text'][ent_span[0]:ent_span[1]])
                    found_spans = find_sub_list(entity_tokens, doc['tokens'])
                    if doc['text'][ent_span[0]:ent_span[1]] not in exist_span.keys():
                        tokenized_spans.extend(found_spans)
                        exist_span[doc['text'][ent_span[0]:ent_span[1]]] = found_spans

                entities = [ent_span for ent_span in tokenized_spans
                            if ent_span[1] < self.cfg.max_length]
            else:
                entities = [ent_span for ent_span in doc['ner']
                            if ent_span[1] < self.cfg.max_length]

            # Iterate over all entity pairs
            for i in range(len(entities)):
                for j in range(i):
                    candidate_pair = entities[i][:2]+entities[j][:2]
                    candidate_pair_w_width = candidate_pair[:2]+[
                        candidate_pair[1]-candidate_pair[0]+1]+candidate_pair[2:]+[
                        candidate_pair[3]-candidate_pair[2]+1]        

                    if self.cfg.use_predicted_entities:
                        if candidate_pair_w_width[2] < self.cfg.max_span_length and candidate_pair_w_width[5] < self.cfg.max_span_length:
                            entity_pairs.append(candidate_pair_w_width)
                            labels.append(
                                self.relation_labels.index("NonRelation"))
                    else:

                        if candidate_pair in positive_pairs and candidate_pair_w_width[2] < self.cfg.max_span_length and candidate_pair_w_width[5] < self.cfg.max_span_length:
                            # add candidate
                            entity_pairs.append(candidate_pair_w_width)
                            # add label
                            label_idx = positive_pairs.index(candidate_pair)
                            labels.append(positive_labels[label_idx])
                        elif random.random() < self.cfg.negative_sample_ratio and candidate_pair_w_width[2] < self.cfg.max_span_length and candidate_pair_w_width[5] < self.cfg.max_span_length:
                            # add negative candidate
                            entity_pairs.append(candidate_pair_w_width)
                            # add negative label
                            labels.append(
                                self.relation_labels.index("NonRelation"))

            global_labels += labels
            docs_w_span_labels.append(
                {**doc, "span_pairs": entity_pairs, "labels": labels})

        return docs_w_span_labels, global_labels

    def encode(self, docs: List[Dict]) -> List[Dict]:

        special_tokens = {}

        def get_special_token(w, special_tokens, unused_tokens=False):
            if w not in special_tokens:
                if unused_tokens:
                    special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
                else:
                    special_tokens[w] = ('<' + w + '>')
            return special_tokens[w] 

        encoded_docs = []
        for doc in docs:

            text = doc['text']

            encodings = self.tokenizer(
                text, padding="max_length", truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
            tokenizer_tokens = self.tokenizer.tokenize(text)
            encoded_docs.append(
                {**doc, "input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "tokens": tokenizer_tokens})

        return encoded_docs

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze(1)

        span_subj = [[span[:3] for span in ex['span_pairs']] for ex in batch]
        span_obj = [[span[3:] for span in ex['span_pairs']] for ex in batch]

        labels = [ex['labels'] for ex in batch]
        labels = torch.tensor(
            [label for sample in labels for label in sample])
        span_mask = torch.tensor([idx for idx, ex in enumerate(
            batch) for span_pairs in ex['span_pairs']])

        texts = []
        tokens = []

        for batch_idx, ex in enumerate(batch):
            texts.append(ex['text'])
            tokens.append(ex['tokens'])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_subj": span_subj,
            "span_obj": span_obj,
            "labels": labels,
            "span_mask": span_mask,
            "texts": texts,
            "tokens": tokens
        }
