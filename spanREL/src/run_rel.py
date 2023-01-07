from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os
import ast
import logging
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
import ipdb
import json
import jsonlines
from models.model import spanREL
from common.utils import get_dataset, get_clearml_file_path, config_clearml_paths, create_inference_dataset
import pandas as pd
from ast import literal_eval

def get_dataloader(split_name: str, cfg) -> Tuple[DataLoader, List, List]:
    data_instance, entity_labels, relation_labels, relation_loss_weights = get_dataset(
        split_name, cfg)

    if split_name == "train":
        return DataLoader(
            data_instance, batch_size=cfg.train_batch_size, collate_fn=data_instance.collate_fn), entity_labels, relation_labels, relation_loss_weights
    else:
        return DataLoader(
            data_instance, batch_size=cfg.eval_batch_size, collate_fn=data_instance.collate_fn), entity_labels, relation_labels


def train(cfg) -> Any:
    train_loader, entity_labels, relation_labels, relation_loss_weights = get_dataloader(
        "train", cfg)
    val_loader, _, _ = get_dataloader("dev", cfg)
    model = spanREL(cfg, num_rel_labels=len(relation_labels),
                    relation_loss_weights=relation_loss_weights)

    callbacks = []

    if cfg.checkpointing:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="./",
            filename="best_entity_lm",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            period=3,
        )
        callbacks.append(checkpoint_callback)

    if cfg.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=8, verbose=False, mode="min")
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        gpus=cfg.gpu, max_epochs=cfg.num_epoch, callbacks=callbacks, check_val_every_n_epoch=cfg.eval_per_epoch)
    trainer.fit(model, train_loader, val_loader)

    return model


def evaluate(cfg, model) -> None:
    test_loader, _, _ = get_dataloader("test", cfg)
    trainer = pl.Trainer(gpus=cfg.gpu)
    trainer.test(model, test_loader)

def predict(cfg, docs: List[Dict]):
    inference_dataset, entity_labels, relation_labels, relation_loss_weights = create_inference_dataset(
        cfg, docs)

    inference_loader = DataLoader(
            inference_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=inference_dataset.collate_fn)

    model = spanREL(cfg, num_rel_labels=len(relation_labels),
                    relation_loss_weights=relation_loss_weights)

    model = model.load_from_checkpoint(
                cfg.rel_trained_model_path, args=cfg, num_rel_labels=len(relation_labels),strict=False)
    
    trainer = pl.Trainer(gpus=cfg.gpu)
    predictions = trainer.predict(model, inference_loader)
    print(predictions)
    predictions_unbatched = []
    for prediction_batch in predictions:
        predictions_unbatched.extend(prediction_batch)
    resultfile = open("/home/shearman/Desktop/work/relation-extraction-module/spanREL/data/predictions.jsonl", "wb")
    writer = jsonlines.Writer(resultfile)
    writer.write_all(predictions_unbatched)

    return predictions_unbatched


@ hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    if cfg.do_train:
        model = train(cfg)

    if cfg.do_eval and model:
        evaluate(cfg, model)
    if cfg.do_predict:
        # print(os.getcwd())
        articles = pd.read_csv("/home/shearman/Desktop/work/relation-extraction-module/spanREL/data/ner_predictions.csv")
        articles['predicted_ner'] = articles.predicted_ner.apply(lambda x: literal_eval(str(x)))   
        texts = articles['text'].tolist()
        ner_predictions = articles['predicted_ner'].tolist()
        docs = [{"text":doc, "predicted_ner": ner_predictions} for doc,ner_predictions in zip(texts, ner_predictions)]
        predict(cfg, docs)

if __name__ == "__main__":
    hydra_main()
