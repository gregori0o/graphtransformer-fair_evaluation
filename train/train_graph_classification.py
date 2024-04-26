"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train.metrics import MAE, accuracy_TU


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_score = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata["feat"].to(device)  # num x feat
        batch_e = batch_graphs.edata["feat"].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(
            batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
        )
        loss = model.loss(batch_scores, batch_targets.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        if model.num_classes == 1:
            epoch_train_score += MAE(batch_scores, batch_targets)
        else:
            epoch_train_score += accuracy_TU(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= iter + 1
    epoch_train_score /= nb_data

    return epoch_loss, epoch_train_score, optimizer


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_score = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"].to(device)
            batch_e = batch_graphs.edata["feat"].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(
                batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
            )
            loss = model.loss(batch_scores, batch_targets.long())
            epoch_test_loss += loss.detach().item()
            if model.num_classes == 1:
                epoch_test_score += MAE(batch_scores, batch_targets)
            else:
                epoch_test_score += accuracy_TU(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= iter + 1
        epoch_test_score /= nb_data

    return epoch_test_loss, epoch_test_score


def full_evaluate_classification(model, device, data_loader, epoch):
    model.eval()
    list_targets = []
    list_predictions = []
    list_probs = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"].to(device)
            batch_e = batch_graphs.edata["feat"].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(
                batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
            )
            list_targets.append(batch_targets.reshape(-1).detach().cpu().numpy())
            list_predictions.append(batch_scores.detach().argmax(dim=1).cpu().numpy())
            list_probs.append(F.softmax(batch_scores, dim=1).detach().cpu().numpy())
    targets = np.concatenate(list_targets)
    predictions = np.concatenate(list_predictions)
    probs = np.concatenate(list_probs)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="micro")
    f1 = f1_score(targets, predictions, average="micro")
    macro_f1 = f1_score(targets, predictions, average="macro")
    # if :
    #     probs = probs[:, 1]
    # roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    unique_values = np.unique(targets)
    if len(unique_values) == model.num_classes:
        if model.num_classes == 2:
            probs = probs[:, 1]
        roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    else:
        roc = 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro f1": macro_f1,
        "roc": roc,
    }
