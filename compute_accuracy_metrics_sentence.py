# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 18:01:20 2022

@author: User
"""
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import segeval
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.metrics import f1_score, precision_score, recall_score


def get_boundaries(boundaries):
    tot_sents = 0
    masses = []
    for boundary in boundaries:
        tot_sents += 1
        if boundary:
            masses.append(tot_sents)
            tot_sents = 0
    return masses

def get_k(ground_truth):
    try:
        k = len(ground_truth)/sum(ground_truth)//2
    except ZeroDivisionError:
        k = 1
    return int(max(k, 1))

def B_measure(boundaries, ground_truth):
    """
    Boundary edit distance-based methods for text segmentation evaluation (Fournier2013)
    """
    boundaries[-1] = 1
    ground_truth[-1] = 1
    k = get_k(ground_truth)
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    cm = segeval.boundary_confusion_matrix(h, t, n_t=4)
    b_precision = float(segeval.precision(cm, classification=1))
    b_recall = float(segeval.recall(cm, classification=1))
    try:
        b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall)
    except ZeroDivisionError:
        b_f1 = 0.0

    b = segeval.boundary_similarity(h, t, n_t=10)

    return float(b_precision), float(b_recall), float(b_f1), float(b)


def sig(x):
    return 1 / (1 + np.exp(-x))


def bootstrap(data, samples=10000):
    if isinstance(data, list):
        data = pd.DataFrame(data)
    boot = []
    for sample in range(samples):
        boot.append(data.sample(len(data), replace=True).mean()[0])
    return boot


data = sys.argv[1]

early_fusion = False

if data == "nonnews":

    with open("NonNewsSentence/NonNews_split.json") as f:
        files = json.load(f)["test"]

    with open("NonNewsSentence/NonNewsSentence/labs_dict.pkl", "rb") as f:
        lab = pickle.load(f)

    prefix = os.path.join("NonNewsSentence", experiment_name)
    prefix_lab = os.path.join("NonNewsSentence", "NonNewsSentence")


elif data == "radionews":

    with open("RadioNewsSentence/RadioNewsSentence/labs_dict.pkl", "rb") as f:
        lab = pickle.load(f)

    with open("RadioNewsSentence/RadioNews_split.json") as f:
        files = json.load(f)["test"]

    prefix = os.path.join("RadioNewsSentence", "UnimodalExperiments")

    if early_fusion:
        folder2 = "ExperimentsMultimodalEarlyFusion"
    else:
        folder2 = "NewLateFusion"

    prefix2 = os.path.join("RadioNewsSentence", folder2)
    prefix_lab = os.path.join("RadioNewsSentence", "RadioNewsSentence")

    prefix3 = os.path.join("RadioNewsSentence", "ExperimentsMultimodalEarlyFusion")

else:
    raise ValueError("Enter one of nonnews or radionews as function argument!")

df = {
    "Precision": [],
    "Precision Confidence": [],
    "Recall": [],
    "Recall Confidence": [],
    "F1": [],
    "F1 Confidence": [],
    "B-F1": [],
    "B-Precision": [],
    "B-Recall": [],
    "B-F1 Confidence": [],
    "B-Precision Confidence": [],
    "B-Recall Confidence": [],
    "embedding": [],
}

all_scores_f1 = {}
all_scores_precision = {}
all_scores_recall = {}

all_scores_bf1 = {}
all_scores_bprecision = {}
all_scores_brecall = {}

op = True

if data=="nonnews":
    encoders = [
    "x-vectors",
    "openl3/_mean_std",
    "radio_news_roberta",
    "radio_news_topseg",
    "radio_news_roberta+radio_news_topseg",
    "x-vectors+openl3/_mean_std",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_roberta",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_topseg",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_roberta+RadioNewsSentence/non_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/non_news_roberta",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/non_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/non_news_roberta+RadioNewsSentence/non_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_roberta",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/non_news_roberta+RadioNewsSentence/non_news_topseg",
]
else:
    encoders = [
    "x-vectors",
    "openl3/_mean_std",
    "radio_news_roberta",
    "radio_news_topseg",
    "radio_news_roberta+radio_news_topseg",
    "x-vectors+openl3/_mean_std",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_roberta",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_topseg",
    "RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_roberta+RadioNewsSentence/radio_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/radio_news_roberta",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/radio_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/radio_news_roberta+RadioNewsSentence/radio_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_roberta",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_topseg",
    "RadioNewsSentence/x-vectors+RadioNewsSentence/openl3/_mean_std+RadioNewsSentence/radio_news_roberta+RadioNewsSentence/radio_news_topseg",
]

for enc in encoders:
    try:
        with open(os.path.join(prefix, "BiLSTM_bs10_" + enc, "all_scores.json")) as f:
            d = json.load(f)
    except FileNotFoundError:
        try:
            with open(os.path.join(prefix2, "BiLSTM_bs10_" + enc, "all_scores.json")) as f:
                d = json.load(f)
        except FileNotFoundError:
            try:
                with open(os.path.join(prefix3, "BiLSTM_bs10_" + enc, "all_scores.json")) as f:
                    d = json.load(f)
            except:
                raise ValueError(f"Directory {enc} not found among the experiments!")

    enc = re.sub("RadioNewsSentence/", "", enc)
    enc = re.sub("NonNewsSentence/", "", enc)

    f1_scores = []
    recall_scores = []
    precision_scores = []
    bf1_scores = []
    bprecision_scores = []
    brecall_scores = []
    for k in files:
        lab_k = os.path.join(k[:-4])

        pred = (sig(np.array(d[k]).reshape(-1)) > 0.5) + 0

        f1_scores.append(f1_score(lab[lab_k][:-1], pred[:-1]))

        recall_scores.append(recall_score(lab[lab_k][:-1], pred[:-1]))

        precision_scores.append(precision_score(lab[lab_k][:-1], pred[:-1]))

        prec, rec, f1, _ = B_measure(pred, lab[lab_k])

        bf1_scores.append(f1)
        bprecision_scores.append(prec)
        brecall_scores.append(rec)

    all_scores_f1[enc] = f1_scores
    all_scores_precision[enc] = precision_scores
    all_scores_recall[enc] = recall_scores

    all_scores_bf1[enc] = bf1_scores
    all_scores_bprecision[enc] = bprecision_scores
    all_scores_brecall[enc] = brecall_scores

    boots = bootstrap(f1_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["F1"].append(np.mean(boots))
    df["F1 Confidence"].append(confidence)

    boots = bootstrap(precision_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["Precision"].append(np.mean(boots))
    df["Precision Confidence"].append(confidence)

    boots = bootstrap(recall_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["Recall"].append(np.mean(boots))
    df["Recall Confidence"].append(confidence)

    boots = bootstrap(bf1_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["B-F1"].append(np.mean(boots))
    df["B-F1 Confidence"].append(confidence)

    boots = bootstrap(bprecision_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["B-Precision"].append(np.mean(boots))
    df["B-Precision Confidence"].append(confidence)

    boots = bootstrap(brecall_scores)

    confidence = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

    df["B-Recall"].append(np.mean(boots))
    df["B-Recall Confidence"].append(confidence)
    
    df["embedding"].append(enc)
    op = False


df = pd.DataFrame(df)

f1_sorted_idx = df["F1"].sort_values(ascending=False).index
precision_sorted_idx = df["Precision"].sort_values(ascending=False).index
recall_sorted_idx = df["Recall"].sort_values(ascending=False).index

bf1_sorted_idx = df["B-F1"].sort_values(ascending=False).index
bprecision_sorted_idx = df["B-Precision"].sort_values(ascending=False).index
brecall_sorted_idx = df["B-Recall"].sort_values(ascending=False).index


def compute_pvalues(scores, sorted_indeces, results_df, b, normal_b, use_ttest=True):
    p1s = np.zeros(len(df))
    p2s = np.zeros(len(df))
    for index, e in enumerate(sorted_indeces[:-1]):
        if not index:
            c = scores[results_df.iloc[e, -1]]
            normal_c = shapiro(c).pvalue > 0.05
        a = scores[df.iloc[e, -1]]

        normal_a = shapiro(a).pvalue > 0.01

        if (normal_a and normal_b) or use_ttest:
            var_a = np.var(a)
            var_b = np.var(b)

            if var_a > var_b:
                var_ratio = var_a / var_b
            else:
                var_ratio = var_b / var_a

            if var_ratio > 4:
                p1s[e] = ttest_ind(a, b, equal_var=False).pvalue
            else:
                p1s[e] = ttest_ind(a, b).pvalue
        else:
            print("Not Normally distributed!")
            p1s[e] = mannwhitneyu(a, b).pvalue

        if (normal_a and normal_c) or use_ttest:
            # print("Normally distributed!")
            var_a = np.var(a)
            var_c = np.var(c)

            if var_a > var_c:
                var_ratio = var_a / var_c
            else:
                var_ratio = var_c / var_a

            if var_ratio > 4:
                p2s[e] = ttest_ind(a, c, equal_var=False, alternative="less").pvalue
            else:
                p2s[e] = ttest_ind(a, c, alternative="less").pvalue
        else:
            print("Not Normally distributed!")
            p2s[e] = mannwhitneyu(a, c).pvalue

    return p1s, p2s


b = all_scores_f1["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
f1_p, f1_p2 = compute_pvalues(all_scores_f1, f1_sorted_idx, df, b, normal_b)

b = all_scores_precision["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
precision_p, precision_p2 = compute_pvalues(
    all_scores_precision, precision_sorted_idx, df, b, normal_b
)

b = all_scores_recall["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
recall_p, recall_p2 = compute_pvalues(
    all_scores_recall, recall_sorted_idx, df, b, normal_b
)

b = all_scores_bf1["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
bf1_p, bf1_p2 = compute_pvalues(all_scores_bf1, bf1_sorted_idx, df, b, normal_b)

b = all_scores_bprecision["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
bprecision_p, bprecision_p2 = compute_pvalues(
    all_scores_bprecision, bprecision_sorted_idx, df, b, normal_b
)

b = all_scores_brecall["radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
brecall_p, brecall_p2 = compute_pvalues(
    all_scores_brecall, brecall_sorted_idx, df, b, normal_b
)

b = all_scores_f1["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
f1_p3, f1_p4 = compute_pvalues(all_scores_f1, f1_sorted_idx, df, b, normal_b)

b = all_scores_precision["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
precision_p3, precision_p4 = compute_pvalues(
    all_scores_precision, precision_sorted_idx, df, b, normal_b
)

b = all_scores_recall["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
recall_p3, recall_p4 = compute_pvalues(
    all_scores_recall, recall_sorted_idx, df, b, normal_b
)

b = all_scores_bf1["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
bf1_p3, bf1_p4 = compute_pvalues(all_scores_bf1, bf1_sorted_idx, df, b, normal_b)

b = all_scores_bprecision["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
bprecision_p3, bprecision_p4 = compute_pvalues(
    all_scores_bprecision, bprecision_sorted_idx, df, b, normal_b
)

b = all_scores_brecall["openl3/_mean_std+radio_news_roberta+radio_news_topseg"]
normal_b = shapiro(b).pvalue > 0.05
brecall_p3, brecall_p4 = compute_pvalues(
    all_scores_brecall, brecall_sorted_idx, df, b, normal_b
)

df["F1 P-value"] = f1_p
df["Precision P-value"] = precision_p
df["Recall P-value"] = recall_p

df["B-F1 P-value"] = bf1_p
df["B-Precision P-value"] = bprecision_p
df["B-Recall P-value"] = brecall_p

df["F1 P-value 2"] = f1_p2
df["Precision P-value 2"] = precision_p2
df["Recall P-value 2"] = recall_p2

df["B-F1 P-value 2"] = bf1_p2
df["B-Precision P-value 2"] = bprecision_p2
df["B-Recall P-value 2"] = brecall_p2

df["F1 P-value3"] = f1_p3
df["Precision P-value3"] = precision_p3
df["Recall P-value3"] = recall_p3

df["B-F1 P-value3"] = bf1_p3
df["B-Precision P-value3"] = bprecision_p3
df["B-Recall P-value3"] = brecall_p3

df["F1 P-value 4"] = f1_p4
df["Precision P-value 4"] = precision_p4
df["Recall P-value 4"] = recall_p4

df["B-F1 P-value 4"] = bf1_p4
df["B-Precision P-value 4"] = bprecision_p4
df["B-Recall P-value 4"] = brecall_p4

df.to_csv(os.path.join(prefix2, "final_result_bilstm.csv"), index=False)