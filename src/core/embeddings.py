import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import pytesseract
from pytesseract import TesseractError
from PIL import Image
import os
import pickle
import re

from src.constants import PYTESSERACT_EXECUTABLE, TESSDATA_DIR_CONFIG, DEFAULT_WIN32_MAX_FILE_OPEN
from src.constants import INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, WORDS_AND_BOXES_DIR
from src.utilities import get_short_names, list_files

pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_EXECUTABLE


def words_from_images(img_list, top_n=1000, box_spacing=5):
    """
    This function finds the top_n words from each image in the img_list. Words are found using pytesseract
    :param img_list: list of image names
    :param top_n: number of top words to be found
    :param box_spacing: spacing around the box
    :return: list of words and list of boxes
    """
    words_list = list()
    boxes_list = list()
    for img_name in img_list:
        print('Computing OCR for:', img_name)
        im = Image.open(img_name)

        ocr_df = pytesseract.image_to_data(im, output_type='data.frame', config=TESSDATA_DIR_CONFIG)
        im.close()

        # text processing
        text = ocr_df.text
        text = text.replace(r'^\s*$', np.nan, regex=True)
        # non_nan indices
        non_nan_idx = [i for i in range(len(text)) if not pd.isnull(text[i])]
        # text = text.dropna().reset_index(drop = True)
        # text = text.replace('^[\[\(\{]', '', regex=True)
        # text = text.replace('[\]\)\}]$', '', regex=True)
        words = list(text)

        # box coordinates
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
            actual_box = [x - box_spacing, y - box_spacing, x + w + box_spacing, y + h + box_spacing]  # we turn it into (left, top, left+widght, top+height) to get the actual box
            boxes.append(actual_box)

        # only keep non_nan_idx
        words = [words[i] for i in range(len(words)) if i in non_nan_idx]
        boxes = [boxes[i] for i in range(len(boxes)) if i in non_nan_idx]

        words_list.append(words[0:top_n])
        boxes_list.append(boxes[0:top_n])

        # ocr_df = ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
        #                                 width_scaled=ocr_df.width * w_scale,
        #                                 top_scaled=ocr_df.top * h_scale,
        #                                 height_scaled=ocr_df.height * h_scale,
        #                                 right_scaled=lambda x: x.left_scaled + x.width_scaled,
        #                                 bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
        #
        # float_cols = ocr_df.select_dtypes('float').columns
        # ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        # ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)  # remove trailing white spaces
        # ocr_df = ocr_df.dropna().reset_index(drop=True)
        # words = list(ocr_df.text)
        # words_list.append(words[0:top_n])

    return [words_list, boxes_list]


# @st.cache(allow_output_mutation = True)
@st.experimental_singleton()
def load_img_transformer_model(model_name):
    return SentenceTransformer(model_name)


# @st.cache(allow_output_mutation = True)
@st.experimental_singleton()
def load_txt_transformer_model(model_name):
    return SentenceTransformer(model_name)


def correct_orientation(img):
    # change landscape images to portrait
    im = Image.open(img)
    if im.width > im.height:
        print('Rotating:', im)
        im = im.rotate(90, expand=True)
        im.save(img)
        im.close()

    # check for orientation in pytesseract
    try:
        print('Reading OSD for:', img)
        orientation = pytesseract.image_to_osd(img)
        orientation = int(re.search('(?<=Rotate: )\d+', orientation).group(0))
        # check for rotated images
        if orientation in [90, 180, 270]:
            im = Image.open(img)
            print('Re-orienting:', img)
            im = im.rotate(360 - orientation, expand=True)
            im.save(img)
            im.close()
    except TesseractError:
        print('Teserract Error for:', img)
        # do nothing
        pass

    # check for RGB. We have trouble finding embeddings for RGBA files
    if im.mode != 'RGB':
        im = Image.open(img)
        print('Making RGB:', img)
        im = im.convert('RGB')
        image_updated = True
        im.save(img)
        im.close()


def load_model(model_type, model_name):
    """Load the appropriate model."""
    print(f"Loading {model_type} model: {model_name}")
    if model_type == "image":
        return load_img_transformer_model(model_name)
    elif model_type == "text":
        return load_txt_transformer_model(model_name)
    else:
        raise ValueError("Unsupported model type")


def save_embeddings(embeddings, output_dir, image_names, file_extension=".emb"):
    """Save embeddings to files."""
    print(f"Saving embeddings to {output_dir}")
    embeddings_np = embeddings.cpu().numpy()
    for i, img_name in enumerate(image_names):
        emb_file = os.path.join(output_dir, img_name + file_extension)
        with open(emb_file, 'wb') as fp:
            pickle.dump(embeddings_np[i:(i + 1)], fp)


def process_in_batches(data, model, batch_size=512):
    """Process data in batches using the provided model."""
    print(f"Computing embeddings for {len(data)} items in batches of {DEFAULT_WIN32_MAX_FILE_OPEN}")
    if len(data) > DEFAULT_WIN32_MAX_FILE_OPEN:
        embeddings = model.encode(data[:DEFAULT_WIN32_MAX_FILE_OPEN],
                                  batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        cnt = 1
        while cnt * DEFAULT_WIN32_MAX_FILE_OPEN < len(data):
            lo_idx = cnt * DEFAULT_WIN32_MAX_FILE_OPEN
            hi_idx = min((cnt + 1) * DEFAULT_WIN32_MAX_FILE_OPEN, len(data))
            tmp_embeddings = model.encode(data[lo_idx:hi_idx],
                                          batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
            embeddings = torch.cat((embeddings, tmp_embeddings))
            cnt += 1
        return embeddings
    else:
        return model.encode(data, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)


def process_embeddings(images, model_type, model_name, output_dir, preprocess_fn=lambda x: x):
    """Generalized function to process embeddings."""
    if not images:
        return

    # Preprocess inputs
    img_names = [os.path.join(INVOICE_DIR, inv) for inv in images]
    data = preprocess_fn(img_names)

    # Load model
    model = load_model(model_type, model_name)

    # Compute embeddings
    embeddings = process_in_batches(data, model)

    # Save embeddings
    save_embeddings(embeddings, output_dir, images)


def preprocess_image(img_names):
    """Preprocess images for model input."""
    return [Image.open(filepath) for filepath in img_names]


def preprocess_text(img_names):
    """Preprocess text data for model input."""
    words_list, boxes_list = words_from_images(img_names)
    corpus = [" ".join(words) for words in words_list]

    # Save words and boxes
    print("Saving word list and boxes")
    for i, img_name in enumerate(img_names):
        wb_file = os.path.join(WORDS_AND_BOXES_DIR, os.path.basename(img_name) + '.wb')
        with open(wb_file, 'wb') as fp:
            pickle.dump([words_list[i], boxes_list[i]], fp)

    return corpus


def compute_embeddings():

    invoice_names = list_files(INVOICE_DIR)
    invoice_names_short = [inv.split('\\')[-1] for inv in invoice_names]

    # image embeddings
    img_embeddings_names = list_files(IMAGE_EMBEDDINGS_DIR)
    img_embeddings_names_short = get_short_names(img_embeddings_names, '.emb')

    # delete embeddings if image is not present
    img_emb_not_inv = [e for e in img_embeddings_names_short if e not in invoice_names_short]
    for e in img_emb_not_inv:
        os.remove(os.path.join(IMAGE_EMBEDDINGS_DIR, e + '.emb'))

    # Now find which invoices have no accompanying embeddings
    inv_not_img_emb = [inv for inv in invoice_names_short if inv not in img_embeddings_names_short]

    # text embeddings
    txt_embeddings_names = list_files(TEXT_EMBEDDINGS_DIR)
    txt_embeddings_names_short = get_short_names(txt_embeddings_names, '.emb')

    # delete embeddings if image is not present
    txt_emb_not_inv = [e for e in txt_embeddings_names_short if e not in invoice_names_short]
    for e in txt_emb_not_inv:
        os.remove(TEXT_EMBEDDINGS_DIR + e + '.emb')

    # Now find which invoices have no accompanying embeddings
    inv_not_txt_emb = [inv for inv in invoice_names_short if inv not in txt_embeddings_names_short]

    # pre-process images
    img_names1 = [os.path.join(INVOICE_DIR, inv) for inv in inv_not_img_emb]
    img_names2 = [os.path.join(INVOICE_DIR, inv) for inv in inv_not_txt_emb]
    img_names = img_names1 + img_names2
    img_names = sorted(list(set(img_names)))

    # correct rotation using pytesseract. Also make the mode RGB
    for img in img_names:
        correct_orientation(img)

    process_embeddings(inv_not_img_emb, "image", "clip-ViT-B-32", IMAGE_EMBEDDINGS_DIR, preprocess_image)
    process_embeddings(inv_not_txt_emb, "text", "all-MiniLM-L6-v2", TEXT_EMBEDDINGS_DIR, preprocess_text)
