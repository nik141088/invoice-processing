import filecmp
import pandas as pd
import numpy as np
from sentence_transformers import util
import torch
import glob
import os
import pickle

from src.constants import INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, DUPLICATES_DIR, DEFAULT_INVOICE_MATCHING_CHOICE
from src.utilities import get_short_names, get_duplicate_files, list_files


def normalize_vector(vec):
    """Normalize a vector to have unit norm."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def load_embeddings(embedding_dir, invoice_names):
    """
    Load and normalize embeddings for all invoices.
    """
    print(f"Loading embeddings from {embedding_dir}")
    embeddings = []
    for invoice in invoice_names:
        emb_file = os.path.join(embedding_dir, invoice + '.emb')
        with open(emb_file, 'rb') as fp:
            tmp = pickle.load(fp)
            embeddings.append(torch.from_numpy(normalize_vector(tmp)))
    return torch.cat(embeddings)


def save_dataframe_to_pickle(df, output_dir, unique_column, file_extension=".dup"):
    """
    Save duplicates DataFrame grouped by a unique column into separate pickle files.
    """
    unique_values = df[unique_column].unique()
    grouped_data = [v for k, v in df.groupby(unique_column)]
    for value, group in zip(unique_values, grouped_data):
        output_file = os.path.join(output_dir, value + file_extension)
        if os.path.exists(output_file):
            with open(output_file, 'rb') as fp:
                existing_data = pickle.load(fp)
            new_data = group[~group.dup.isin(existing_data.dup.values)]
            final_data = pd.concat([existing_data, new_data])
        else:
            final_data = group
        with open(output_file, 'wb') as fp:
            pickle.dump(final_data, fp)


def compute_filecmp_duplicates(invoice_names):
    """
    Compute duplicates using file comparison.
    """
    print('Computing filecmp duplicates')
    filecmp_dups = [
        [org, dup, filecmp.cmp(os.path.join(INVOICE_DIR, org), os.path.join(INVOICE_DIR, dup), shallow=False)]
        for i, org in enumerate(invoice_names)
        for j, dup in enumerate(invoice_names) if i < j
    ]
    df = pd.DataFrame(filecmp_dups, columns=['org', 'dup', 'filecmp'])
    return reorder_dataframe_columns(df, "filecmp")


def compute_similarity_duplicates(embeddings, invoice_names, column_name):
    """
    Compute duplicates using similarity scores for embeddings.
    """
    print(f'Computing {column_name} duplicates')
    scores = util.paraphrase_mining_embeddings(embeddings)
    results = []
    for score, idx1, idx2 in scores:
        score = min(score, 1.0)
        org = invoice_names[idx1].split('\\')[-1]
        dup = invoice_names[idx2].split('\\')[-1]
        results.append([org, dup, score])
    df = pd.DataFrame(results, columns=['org', 'dup', column_name])
    return reorder_dataframe_columns(df, column_name)


def reorder_dataframe_columns(df, score_column):
    """
    Ensure 'org' is always lexicographically less than 'dup' and reorder columns.
    """
    right = df[df.org < df.dup]
    wrong = df[df.org > df.dup]
    wrong = wrong.rename(columns={"org": "dup", "dup": "org"})
    df = pd.concat([right, wrong], ignore_index=True)
    return df


def merge_and_finalize_duplicates(filecmp_dups, img_dups, txt_dups):
    """
    Merge all duplicates into a single DataFrame with additional columns.
    """
    duplicates = pd.merge(filecmp_dups, img_dups, on=['org', 'dup'], how='outer')
    duplicates = pd.merge(duplicates, txt_dups, on=['org', 'dup'], how='outer')
    duplicates = duplicates.sort_values(["org", "dup"])
    duplicates["action"] = DEFAULT_INVOICE_MATCHING_CHOICE
    duplicates["seen"] = "not_seen"
    return duplicates


def compute_missing_combinations(invoice_names_short, existing_dups, img_emb, txt_emb):
    """
    Compute duplicates for missing combinations efficiently.
    """
    print('Computing duplicates ONLY for missing combinations!')
    inv_not_dup = [inv for inv in invoice_names_short if inv not in existing_dups]
    results = []
    for i, org in enumerate(inv_not_dup):
        for j, dup in enumerate(existing_dups):
            img_score = util.paraphrase_mining_embeddings(torch.cat((img_emb[i:(i + 1)], img_emb[j:(j + 1)])))[0][0]
            txt_score = util.paraphrase_mining_embeddings(torch.cat((txt_emb[i:(i + 1)], txt_emb[j:(j + 1)])))[0][0]
            filecmp_score = filecmp.cmp(os.path.join(INVOICE_DIR, org), os.path.join(INVOICE_DIR, dup), shallow=False)
            results.append([org, dup, filecmp_score, img_score, txt_score, DEFAULT_INVOICE_MATCHING_CHOICE, "not_seen"])
    if results:
        return pd.DataFrame(results, columns=['org', 'dup', 'filecmp', 'img_score', 'txt_score', 'action', 'seen'])
    # return empty df with the above columns
    return pd.DataFrame(columns=['org', 'dup', 'filecmp', 'img_score', 'txt_score', 'action', 'seen'])


def compute_duplicates():
    invoice_names = list_files(INVOICE_DIR)
    invoice_names_short = [os.path.basename(inv) for inv in invoice_names]
    duplicate_names = get_duplicate_files()
    existing_dups = get_short_names(duplicate_names, ".dup")

    # Cleanup orphan duplicate files
    print('Removing orphan duplicate files')
    for d in set(existing_dups) - set(invoice_names_short):
        os.remove(os.path.join(DUPLICATES_DIR, d + '.dup'))

    # Load embeddings
    img_emb = load_embeddings(IMAGE_EMBEDDINGS_DIR, invoice_names_short)
    txt_emb = load_embeddings(TEXT_EMBEDDINGS_DIR, invoice_names_short)

    if not existing_dups:
        filecmp_dups = compute_filecmp_duplicates(invoice_names_short)
        img_dups = compute_similarity_duplicates(img_emb, invoice_names_short, 'img_score')
        txt_dups = compute_similarity_duplicates(txt_emb, invoice_names_short, 'txt_score')
        duplicates = merge_and_finalize_duplicates(filecmp_dups, img_dups, txt_dups)
    else:
        duplicates = compute_missing_combinations(invoice_names_short, existing_dups, img_emb, txt_emb)

    if duplicates is not None:
        print('Saving duplicates')
        save_dataframe_to_pickle(duplicates, DUPLICATES_DIR, "org")
