import filecmp
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pytesseract
from pytesseract import TesseractError
from PIL import Image, ImageDraw
import glob
import win32file
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import os
import pickle
import math
import time
import re

# pytesseract config on my computer
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r"C:\Program Files\poppler-21.03.0\Library\bin";
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


# load lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Find more emojis here: /https://www.webfx.com/tools/emoji-cheat-sheet
st.set_page_config(page_title="Invoice-Processing (with Transformers)", page_icon = ":muscle:", layout="wide")

# load css and animation
local_css("app/style/style.css")
lottie_invoice = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_g07hYi.json")



# all directories are defined relative to root_dir. If in future the app is moved to a different folder, then only root_dir needs to be changed!
root_dir = "C:\\Users\\nikhi\\invoice-processing\\data\\"
if not os.path.exists(root_dir):
    assert False

# all the invoices are present in invoice_dir.
# We use img_embeddings_dir to create image imbeddings for all the images
# all the duplicates are saved in duplicates_dir. Duplicates are saved wrt each invoice.
# inv_done_dir is used to cache invoices as and when all their duplicates are updated. This is required so that one can pick up dup checking from where they left
invoice_dir = root_dir + "sample_invoices\\"
img_embeddings_dir = root_dir + ".img_emb\\"
txt_embeddings_dir = root_dir + ".txt_emb\\"
words_and_boxes_dir = root_dir + ".words_and_boxes\\"
duplicates_dir = root_dir + ".duplicates\\"
inv_done_dir = root_dir + ".invDone\\"

# prev_loop_iter_file is a keyed stack, i.e. its a stack which also checks for entered key.
# This is used to keep track of sequence of invoices that the user has checked. We use this to go back to earlier invoice!
prev_loop_iter_file = root_dir + '.prev_i'
if not os.path.exists(prev_loop_iter_file):
    with open(prev_loop_iter_file, 'wb') as fp:
        curr_inv = ''
        curr_data = list()
        pickle.dump([curr_inv, curr_data], fp)



# it's very unlikely that two copies of the same invoice will have a similarity less than this
IMG_SIMILARITY_CUTOFF = 0.9
TXT_SIMILARITY_CUTOFF = 0.7
ABS_TOL = 1e-5


inv_matching_opts = ["No Action yet!", "Different", "Pure Duplicate", "Near Duplicate", "More Action"]


# this function clears the stack and its key
def loop_counter_stack_clear():
    curr_inv = ''
    curr_data = list()
    with open(prev_loop_iter_file, 'wb') as fp:
        pickle.dump([curr_inv, curr_data], fp)

# If the key matches with stored key, then one element is popped out of stack.
# If key doesn't match then stack is cleared
def loop_counter_stack_pop(inv):
    with open(prev_loop_iter_file, 'rb') as fp:
        curr_inv, curr_data = pickle.load(fp)
    if inv != '' and curr_inv == inv:
        # all good
        if len(curr_data) == 0:
            return None
        else:
            # save data after popping
            curr_inv = inv
            ret_data = curr_data[-1]
            curr_data = curr_data[:-1]
            with open(prev_loop_iter_file, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)
            return ret_data
    else:
        # clear stack
        loop_counter_stack_clear()
        return None


# If pushing for the first time, then the key and the data are used to create a new stack
# If pushing to an existing stck then supplied key must match with the existing key.
# If they don't then stack is first claered and new key, data pair is stored as the first element
def loop_counter_stack_push(inv, new_data):
    # read current data
    with open(prev_loop_iter_file, 'rb') as fp:
        curr_inv, curr_data = pickle.load(fp)
    # if pushing for the first time
    if curr_inv == '':
        curr_inv = inv
        curr_data = list()
        if curr_inv != None and curr_inv != '':
            curr_data.append(new_data)
        else:
            curr_inv = ''
            curr_data = list()
        with open(prev_loop_iter_file, 'wb') as fp:
            pickle.dump([curr_inv, curr_data], fp)
    else:
        # if pushing to an existing stack, then check for invoice name
        if curr_inv == inv:
            # all good
            curr_inv = inv
            curr_data.append(new_data)
            with open(prev_loop_iter_file, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)
        else:
            # clear stack
            loop_counter_stack_clear()
            # now write with new invoice
            curr_inv = inv
            curr_data = list()
            curr_data.append(new_data)
            with open(prev_loop_iter_file, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)



# @st.cache(allow_output_mutation = True)
@st.experimental_singleton()
def load_img_transformer_model(model_name):
    return SentenceTransformer(model_name)


# @st.cache(allow_output_mutation = True)
@st.experimental_singleton()
def load_txt_transformer_model(model_name):
    return SentenceTransformer(model_name)



# @st.cache
@st.experimental_memo
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# @st.cache
@st.experimental_memo
def download_all_image_duplicates(ret_csv = True):
    dups = list(glob.glob(duplicates_dir + '*'))
    dt_list = [pickle.load(open(dup_file, 'rb')) for dup_file in dups]
    df = pd.concat(dt_list)
    df = df.sort_values(["org", "dup"])
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if ret_csv:
        return df.to_csv().encode('utf-8')
    else:
        return df






# find OCR words in each of the image
def words_from_images(img_list, top_N = 1000, BOX_SPACING = 5):
    words_list = list()
    boxes_list = list()
    for img_name in img_list:
        print('Computing OCR for:', img_name)
        im = Image.open(img_name)

        ocr_df = pytesseract.image_to_data(im, output_type='data.frame', config=tessdata_dir_config)
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
            actual_box = [x - BOX_SPACING, y - BOX_SPACING, x + w + BOX_SPACING, y + h + BOX_SPACING]  # we turn it into (left, top, left+widght, top+height) to get the actual box
            boxes.append(actual_box)

        # only keep non_nan_idx
        words = [words[i] for i in range(len(words)) if i in non_nan_idx]
        boxes = [boxes[i] for i in range(len(boxes)) if i in non_nan_idx]

        words_list.append(words[0:top_N])
        boxes_list.append(boxes[0:top_N])

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
        # words_list.append(words[0:top_N])

    return [words_list, boxes_list]






# create dirs if they don't exist
for dir in [invoice_dir, img_embeddings_dir, txt_embeddings_dir, words_and_boxes_dir, duplicates_dir, inv_done_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)





# normalize image vector
# see https://stackoverflow.com/questions/41387000/cosine-similarity-of-word2vec-more-than-1
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec = vec / norm
    return vec





def compute_embeddings():

    invoice_names = list(glob.glob(invoice_dir + '*'))
    invoice_names_short = [inv.split('\\')[-1] for inv in invoice_names]

    # image embeddings
    img_embeddings_names = list(glob.glob(img_embeddings_dir + '*'))
    img_embeddings_names_short = [emb.split('\\')[-1].rstrip('.emb') for emb in img_embeddings_names]

    # delete embeddings if image is not present
    img_emb_not_inv = [e for e in img_embeddings_names_short if e not in invoice_names_short]
    for e in img_emb_not_inv:
        os.remove(img_embeddings_dir + e + '.emb')

    # Now find which invoices have no accompanying embeddings
    inv_not_img_emb = [inv for inv in invoice_names_short if inv not in img_embeddings_names_short]


    # text embeddings
    txt_embeddings_names = list(glob.glob(txt_embeddings_dir + '*'))
    txt_embeddings_names_short = [emb.split('\\')[-1].rstrip('.emb') for emb in txt_embeddings_names]

    # delete embeddings if image is not present
    txt_emb_not_inv = [e for e in txt_embeddings_names_short if e not in invoice_names_short]
    for e in txt_emb_not_inv:
        os.remove(txt_embeddings_dir + e + '.emb')

    # Now find which invoices have no accompanying embeddings
    inv_not_txt_emb = [inv for inv in invoice_names_short if inv not in txt_embeddings_names_short]


    # pre-process images
    img_names1 = [invoice_dir + inv for inv in inv_not_img_emb]
    img_names2 = [invoice_dir + inv for inv in inv_not_txt_emb]
    img_names = img_names1 + img_names2
    img_names = sorted(list(set(img_names)))
    # correct rotation using pytesseract. Also make the mode RGB
    for img in img_names:

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
                im = im.rotate(360-orientation, expand=True)
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




    if len(inv_not_img_emb) > 0:

        img_names = [invoice_dir + inv for inv in inv_not_img_emb]

        # Now search using transformers (Loading from a cached function)
        img_model = load_img_transformer_model('clip-ViT-B-32')
        # run img_model.encode in batches of default_max_file_open and stack tensors together
        default_max_file_open = win32file._getmaxstdio()
        print('Computing image embeddings')
        if len(img_names) > default_max_file_open:
            img_emb = img_model.encode([Image.open(filepath) for filepath in img_names[0:default_max_file_open]],
                                   batch_size=512, convert_to_tensor=True, show_progress_bar=True)
            cnt = 1
            while cnt * default_max_file_open < len(img_names):
                print(cnt)
                lo_idx = cnt * default_max_file_open
                hi_idx = (cnt + 1) * default_max_file_open
                hi_idx = min(hi_idx, len(img_names))
                img_emb_tmp = img_model.encode([Image.open(filepath) for filepath in img_names[lo_idx:hi_idx]],
                                           batch_size=512, convert_to_tensor=True, show_progress_bar=True)
                img_emb = torch.cat((img_emb, img_emb_tmp))
                cnt = cnt + 1
            del img_emb_tmp
        else:
            img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=512, convert_to_tensor=True, show_progress_bar=True)

        # save embeddings
        print('Saving image embeddings')
        img_emb_cpy = img_emb.cpu().numpy()
        for i in range(img_emb_cpy.shape[0]):
            emb_file = img_embeddings_dir + inv_not_img_emb[i] + '.emb'
            with open(emb_file, 'wb') as fp:
                pickle.dump(img_emb_cpy[i:(i+1)], fp)

        del img_emb, img_emb_cpy


    if len(inv_not_txt_emb) > 0:

        img_names = [invoice_dir + inv for inv in inv_not_txt_emb]

        words_list, boxes_list = words_from_images(img_names)
        # save for future use:
        print('Saving word list and boxes')
        for i in range(len(inv_not_txt_emb)):
            wb_file = words_and_boxes_dir + inv_not_txt_emb[i] + '.wb'
            with open(wb_file, 'wb') as fp:
                pickle.dump([words_list[i], boxes_list[i]], fp)


        # Now search using transformers (Loading from a cached function)
        txt_model = load_txt_transformer_model('all-MiniLM-L6-v2')
        corpus = [" ".join(l) for l in words_list]
        # run txt_model.encode in batches of default_max_file_open and stack tensors together
        default_max_file_open = win32file._getmaxstdio()
        print('Computing text embeddings')
        if len(img_names) > default_max_file_open:
            txt_emb = txt_model.encode(corpus[0:default_max_file_open],
                                       batch_size=512, convert_to_tensor=True, show_progress_bar=True)
            cnt = 1
            while cnt * default_max_file_open < len(img_names):
                print(cnt)
                lo_idx = cnt * default_max_file_open
                hi_idx = (cnt + 1) * default_max_file_open
                hi_idx = min(hi_idx, len(img_names))
                txt_emb_tmp = txt_model.encode(corpus[lo_idx:hi_idx],
                                               batch_size=512, convert_to_tensor=True, show_progress_bar=True)
                txt_emb = torch.cat((txt_emb, txt_emb_tmp))
                cnt = cnt + 1
            del txt_emb_tmp
        else:
            txt_emb = txt_model.encode(corpus, batch_size=512, convert_to_tensor=True, show_progress_bar=True)

        # save embeddings
        print('Saving text embeddings')
        txt_emb_cpy = txt_emb.cpu().numpy()
        for i in range(txt_emb_cpy.shape[0]):
            emb_file = txt_embeddings_dir + inv_not_txt_emb[i] + '.emb'
            with open(emb_file, 'wb') as fp:
                pickle.dump(txt_emb_cpy[i:(i+1)], fp)

        del txt_emb, txt_emb_cpy




def compute_duplicates():

    invoice_names = list(glob.glob(invoice_dir + '*'))
    invoice_names_short = [inv.split('\\')[-1] for inv in invoice_names]

    duplicates_names = list(glob.glob(duplicates_dir + '*'))
    duplicates_names_short = [dup_names.split('\\')[-1].rstrip('.dup') for dup_names in duplicates_names]

    # delete duplicates if image is not present
    dup_not_inv = [d for d in duplicates_names_short if d not in invoice_names_short]
    for d in dup_not_inv:
        os.remove(duplicates_dir + d)

    # load all embeddings: this is needed because duplicates can't be computed without loading all image embeddings
    # Also note that addition of even a single image will lead to changes in all duplicate files
    
    # image embeddings
    print('Loading image embeddings!')
    # load first embedding
    img_emb_file = img_embeddings_dir + invoice_names_short[0] + '.emb'
    with open(img_emb_file, 'rb') as fp:
        tmp = pickle.load(fp)
        img_emb_cpu = torch.from_numpy(tmp)
        img_emb_cpu = normalize_vector(img_emb_cpu)
    # now load others and cat to img_emb
    for inv in invoice_names_short[1:]:
        img_emb_file = img_embeddings_dir + inv + '.emb'
        with open(img_emb_file, 'rb') as fp:
            tmp = pickle.load(fp)
            img_emb_tmp = torch.from_numpy(tmp)
            img_emb_tmp = normalize_vector(img_emb_tmp)
        img_emb_cpu = torch.cat((img_emb_cpu, img_emb_tmp))
    # img_emb = img_emb_cpu.cuda()
    img_emb = img_emb_cpu
    del img_emb_tmp, img_emb_cpu
    
    # text embeddings
    print('Loading text embeddings!')
    # load first embedding
    txt_emb_file = txt_embeddings_dir + invoice_names_short[0] + '.emb'
    with open(txt_emb_file, 'rb') as fp:
        tmp = pickle.load(fp)
        txt_emb_cpu = torch.from_numpy(tmp)
        txt_emb_cpu = normalize_vector(txt_emb_cpu)
    # now load others and cat to txt_emb
    for inv in invoice_names_short[1:]:
        txt_emb_file = txt_embeddings_dir + inv + '.emb'
        with open(txt_emb_file, 'rb') as fp:
            tmp = pickle.load(fp)
            txt_emb_tmp = torch.from_numpy(tmp)
            txt_emb_tmp = normalize_vector(txt_emb_tmp)
        txt_emb_cpu = torch.cat((txt_emb_cpu, txt_emb_tmp))
    # txt_emb = txt_emb_cpu.cuda()
    txt_emb = txt_emb_cpu
    del txt_emb_tmp, txt_emb_cpu


    duplicates = None

    # if the duplicates folder is empty then compute all at once! Otherwise do it selectively.
    if len(duplicates_names_short) == 0:

        print('Computing duplicates for all combinations!')

        # first find pure filecmp duplicates (for all pairs of invoice_names)
        print('filecmp duplicates')
        filecmp_dups = list()
        for i, org in enumerate(invoice_names_short):
            for j, dup in enumerate(invoice_names_short):
                if i < j:
                    cmp = filecmp.cmp(invoice_dir + org, invoice_dir + dup, shallow = False)
                    filecmp_dups.append([org, dup, cmp])
        filecmp_dups = pd.DataFrame(filecmp_dups)
        filecmp_dups.columns = ['org', 'dup', 'filecmp']
        # make sure org is always less than dup (string comparision)
        right = filecmp_dups[filecmp_dups.org < filecmp_dups.dup]
        wrong = filecmp_dups[filecmp_dups.org > filecmp_dups.dup]
        wrong = wrong[['dup', 'org', 'filecmp']]
        wrong.columns = ['org', 'dup', 'filecmp']
        filecmp_dups = right.append(wrong, ignore_index=True)


        # image duplicates
        print('image duplicates')
        img_dups = util.paraphrase_mining_embeddings(img_emb)
        for i in range(len(img_dups)):
            img1 = invoice_names[img_dups[i][1]].split('\\')[-1]
            img2 = invoice_names[img_dups[i][2]].split('\\')[-1]
            # also set duplicate score greater than 1.0 to 1.0
            img_dups[i][0] = min(img_dups[i][0], 1.0)
            # change image number to image name
            img_dups[i][1] = img1
            img_dups[i][2] = img2
        # convert to dataframe
        img_dups = pd.DataFrame(img_dups)
        img_dups.columns = ['img_score', 'org', 'dup']
        # reorder columns
        img_dups = img_dups[["org", "dup", "img_score"]]
        # make sure org is always less than dup (string comparision)
        right = img_dups[img_dups.org < img_dups.dup]
        wrong = img_dups[img_dups.org > img_dups.dup]
        wrong = wrong[['dup', 'org', 'img_score']]
        wrong.columns = ['org', 'dup', 'img_score']
        img_dups = right.append(wrong, ignore_index=True)

        
        # text duplicates
        print('text duplicates')
        txt_dups = util.paraphrase_mining_embeddings(txt_emb)
        for i in range(len(txt_dups)):
            img1 = invoice_names[txt_dups[i][1]].split('\\')[-1]
            img2 = invoice_names[txt_dups[i][2]].split('\\')[-1]
            # also set duplicate score greater than 1.0 to 1.0
            txt_dups[i][0] = min(txt_dups[i][0], 1.0)
            # change image number to image name
            txt_dups[i][1] = img1
            txt_dups[i][2] = img2
        # convert to dataframe
        txt_dups = pd.DataFrame(txt_dups)
        txt_dups.columns = ['txt_score', 'org', 'dup']
        # reorder columns
        txt_dups = txt_dups[["org", "dup", "txt_score"]]
        # make sure org is always less than dup (string comparision)
        right = txt_dups[txt_dups.org < txt_dups.dup]
        wrong = txt_dups[txt_dups.org > txt_dups.dup]
        wrong = wrong[['dup', 'org', 'txt_score']]
        wrong.columns = ['org', 'dup', 'txt_score']
        txt_dups = right.append(wrong, ignore_index=True)

        
        # merge all together
        duplicates = pd.merge(filecmp_dups, img_dups, on = ['org', 'dup'], how = 'outer')
        duplicates = pd.merge(duplicates,   txt_dups, on = ['org', 'dup'], how = 'outer')


        # sort
        duplicates = duplicates.sort_values(["org", "dup"])

        # add default action
        duplicates["action"] = inv_matching_opts[0]

        # add another column to reflect whether next invoice button has been clicked for that
        duplicates["seen"] = "not_seen"

    else:

        # the efficient way to compute duplicates is to look at all inv_not_dup and find duplicates of every invoice in inv_not_dup against all invoices in invoice_names_short
        print('Computing duplicates ONLY for missing combinations!')


        inv_not_dup = [inv for inv in invoice_names_short if inv not in duplicates_names_short]
        inv_not_dup_idx = [invoice_names_short.index(new_inv) for new_inv in inv_not_dup]
        existing_dups = duplicates_names_short
        existing_dups_idx = [invoice_names_short.index(new_inv) for new_inv in existing_dups]

        dup_list = list()
        for i in inv_not_dup_idx:
            for j in existing_dups_idx:
                ext_inv = invoice_names_short[min(i, j)]
                new_inv = invoice_names_short[max(i, j)]
                # img dup
                img_emb_ij  = torch.cat((img_emb[i:(i + 1)], img_emb[j:(j + 1)]))
                img_dups_ij = util.paraphrase_mining_embeddings(img_emb_ij)[0][0]
                img_dups_ij = min(img_dups_ij, 1.0)
                # text dup
                txt_emb_ij  = torch.cat((txt_emb[i:(i + 1)], txt_emb[j:(j + 1)]))
                txt_dups_ij = util.paraphrase_mining_embeddings(txt_emb_ij)[0][0]
                txt_dups_ij = min(txt_dups_ij, 1.0)
                # file comparision dup
                filecmp_dups_ij = filecmp.cmp(invoice_dir + ext_inv, invoice_dir + new_inv, shallow = False)
                # add to dt_list
                dup_list.append([ext_inv, new_inv, filecmp_dups_ij, img_dups_ij, txt_dups_ij, inv_matching_opts[0], "not_seen"])
            # add i to existing_dups_idx
            existing_dups_idx.append(i)

        # make data frame from dup_list
        if len(dup_list) > 0:
            duplicates = pd.DataFrame(dup_list)
            duplicates.columns = ['org', 'dup', 'filecmp', 'img_score', 'txt_score', 'action', 'seen']
            # sort
            duplicates = duplicates.sort_values(["org", "dup"])


    # save duplicates one by one
    if duplicates is not None:
        print('Saving duplicates')
        dup_org = duplicates["org"].unique()
        dt_list = [v for k, v in duplicates.groupby("org")]
        for i in range(len(dup_org)):
            dup_file = duplicates_dir + dup_org[i] + '.dup'
            dt = dt_list[i]
            # read an existing file and append to it the new information, else store it afresh
            if os.path.exists(dup_file):
                with open(dup_file, 'rb') as fp:
                    read_dt = pickle.load(fp)
                new_dt = dt[~dt.dup.isin(read_dt.dup.values)]
                final_dt = pd.concat([read_dt, new_dt])
                with open(dup_file, 'wb') as fp:
                    pickle.dump(final_dt, fp)
            else:
                with open(dup_file, 'wb') as fp:
                    pickle.dump(dt, fp)





# ---- header ----
with st.container():

    left_column, right_column = st.columns((1,1))

    with left_column:
        st.title(":muscle: Invoice-Processing (with Transformers)")
        st.subheader("Process Duplicate Invoices!")
        st.write(
            "This app lets you detect and process duplicate invoices! It also takes care of rotated images!"
        )

        st.write("##")
        st.subheader("Let's Go!")
        show_duplicates = st.checkbox("Show Duplicates")
        data_has_changed = st.checkbox("Data has changed?")
        refresh_db = st.checkbox("Re-estimate the model?")

    with right_column:
        st_lottie(lottie_invoice, height=400, key="invoice")
        # pass


if refresh_db:
    refresh_db_confirm = st.checkbox("Confirm? Everything will be computed afresh!")
    if refresh_db_confirm:
        # reset refresh_db and refresh_db_confirm
        refresh_db = False
        refresh_db_confirm = False

        # delete all image embeddings
        img_embeddings_names = list(glob.glob(img_embeddings_dir + '*'))
        for e in img_embeddings_names:
            os.remove(e)

        # delete all text embeddings
        txt_embeddings_names = list(glob.glob(txt_embeddings_dir + '*'))
        for e in txt_embeddings_names:
            os.remove(e)

        # delete all words and boxes embeddings
        wb_names = list(glob.glob(words_and_boxes_dir + '*'))
        for w in wb_names:
            os.remove(w)

        # delete all duplicates data
        duplicates_names = list(glob.glob(duplicates_dir + '*'))
        for d in duplicates_names:
            os.remove(d)

        # delete all invoice dones
        done_names = list(glob.glob(inv_done_dir + '*'))
        for d in done_names:
            os.remove(d)

        # remove prev_loop_iter_file
        os.remove(prev_loop_iter_file)

        # compute embeddings and duplicates
        compute_embeddings()
        compute_duplicates()

        # display done message
        st.subheader('Model computation done!')



if data_has_changed:

    # compute embeddings and duplicates
    compute_embeddings()
    compute_duplicates()

    # delete all invoice dones
    done_names = list(glob.glob(inv_done_dir + '*'))
    for d in done_names:
        os.remove(d)

    # remove prev_loop_iter_file
    os.remove(prev_loop_iter_file)

    # display done message
    st.subheader('Model check done!')





if show_duplicates:

    with st.container():

        # recompute duplicate names
        duplicates_names = list(glob.glob(duplicates_dir + '*'))
        duplicates_names_short = [dup_names.split('\\')[-1].rstrip('.dup') for dup_names in duplicates_names]

        if len(duplicates_names_short) == 0:

            compute_embeddings()
            compute_duplicates()

            # delete all invoice dones
            done_names = list(glob.glob(inv_done_dir + '*'))
            for d in done_names:
                os.remove(d)

            # remove prev_loop_iter_file
            os.remove(prev_loop_iter_file)

            # recompute duplicate_names
            duplicates_names = list(glob.glob(duplicates_dir + '*'))
            duplicates_names_short = [dup_names.split('\\')[-1].rstrip('.dup') for dup_names in duplicates_names]


        # invoice done files
        done_names = list(glob.glob(inv_done_dir + '*'))
        done_names_short = [dn.split('\\')[-1].rstrip('.done') for dn in done_names]

        # which invoices out of duplicates_name_shorts are remaining after accounting for done_names_short
        rem_names_short = [d for d in duplicates_names_short if d not in done_names_short]
        if len(rem_names_short) == 0:
            # we are done! stop the app!
            st.subheader('***All invoices done!***')
            st.stop()

        st.write("---")
        st.title("Deep Dive!")

        left_column, middle_column, right_column = st.columns((2,1,2))

        with left_column:
            img_threshold = st.slider('Select image similarity cutoff (in %):',
                                      min_value=round(100.0*IMG_SIMILARITY_CUTOFF, 0),
                                      max_value=100.0,
                                      value=round(0.5*100*IMG_SIMILARITY_CUTOFF + 0.5*100.0, 0),
                                      step=1.0)
            txt_threshold = st.slider('Select text similarity cutoff (in %):',
                                      min_value=round(100.0*TXT_SIMILARITY_CUTOFF, 0),
                                      max_value=100.0,
                                      value=round(0.5*100*TXT_SIMILARITY_CUTOFF + 0.5*100.0, 0),
                                      step=1.0)
            # selected_inv = st.selectbox('Select Invoice:', rem_names_short)
            # Alternatively, always choose first element of rem_names_short as selected_inv and remove selectbox
            # we will also need to remove these buttons: redo_selected_invoice, reset_selected_invoice, st.download_button (for invoice level download)
            selected_inv = rem_names_short[0]
            st.write('***Selected Invoice:*** ' + selected_inv)
            skip_actioned_items = st.checkbox("Don't show updated pairs?", value = False)
            skip_seen_items = st.checkbox("Don't show already seen pairs?", value = True)
            hide_pure_duplicates = st.checkbox("Hide pairs with score of 100.0?", value = False)
            show_image_diff = st.checkbox("Highlight text differences?", value = False)


        with middle_column:
            redo_selected_invoice = st.button('Redo selected invoice?')
            if redo_selected_invoice:
                # read duplicate table corresponding to selected invoice
                dup_file = duplicates_dir + selected_inv + '.dup'
                with open(dup_file, 'rb') as fp:
                    dt = pickle.load(fp)
                # set seen to unseen
                dt["seen"] = "not_seen"
                # save it
                with open(dup_file, 'wb') as fp:
                    pickle.dump(dt, fp)
                # push current loop iter
                loop_counter_stack_clear()


            reset_selected_invoice = st.button('Reset selected invoice?')
            if reset_selected_invoice:
                # read duplicate table corresponding to selected invoice
                dup_file = duplicates_dir + selected_inv + '.dup'
                with open(dup_file, 'rb') as fp:
                    dt = pickle.load(fp)
                # set seen to unseen
                dt["action"] = inv_matching_opts[0]
                dt["seen"] = "not_seen"
                # save it
                with open(dup_file, 'wb') as fp:
                    pickle.dump(dt, fp)
                # push current loop iter
                loop_counter_stack_clear()


        # read duplicate table corresponding to selected invoice
        with open(duplicates_dir + selected_inv + '.dup', 'rb') as fp:
            dt = pickle.load(fp)
        dt = dt.sort_values(["filecmp", "img_score", "txt_score"], ascending = [False, False, False])

        # list of possible duplicate invoices
        filter_dt = dt[(dt.img_score >= img_threshold/100.0) & (dt.txt_score >= txt_threshold/100.0)]
        if skip_actioned_items:
            filter_dt = filter_dt[(filter_dt.action == inv_matching_opts[0])]
        if skip_seen_items:
            filter_dt = filter_dt[(filter_dt.seen == "not_seen")]
        if hide_pure_duplicates:
            filter_dt = filter_dt[(filter_dt.filecmp == False)]
            filter_dt = filter_dt[(filter_dt.img_score <= 1.0 - ABS_TOL) | (filter_dt.txt_score <= 1.0 - ABS_TOL)]

        # choose dup invoice if there is atleast one entry present
        if filter_dt.shape[0] > 0:
            selected_inv_dups = dt.dup.to_numpy()
        else:
            selected_inv_dups = []
            # mark this inv as done
            left_column, middle_column, right_column = st.columns((1, 1, 1))
            with middle_column:
                st.subheader('***No more duplicates for this invoice!***')
                mark_this_inv_done = st.button("Mark this invoice done? This can't be changed.")
                if mark_this_inv_done:
                    with open(inv_done_dir + selected_inv + '.done', 'wb') as fp:
                        pickle.dump(None, fp)
                    # clear loop iteration stack
                    loop_counter_stack_clear()
                    # rerun
                    st.experimental_rerun()



        for i in range(len(selected_inv_dups)):

            curr_df = dt[dt.dup == selected_inv_dups[i]];

            curr_resp  = curr_df.action.to_numpy()[0]
            curr_seen  = curr_df.seen.to_numpy()[0]
            curr_filecmp = curr_df.filecmp.to_numpy()[0]
            curr_img_score = curr_df.img_score.to_numpy()[0]
            curr_txt_score = curr_df.txt_score.to_numpy()[0]

            if skip_seen_items and curr_seen == "seen":
                continue

            if skip_actioned_items and curr_resp != inv_matching_opts[0]:
                continue

            if hide_pure_duplicates and ((curr_filecmp is True) or
                                         (math.isclose(curr_img_score, 1.0, abs_tol = ABS_TOL) and
                                         math.isclose(curr_txt_score, 1.0, abs_tol = ABS_TOL))):
                continue

            if (curr_img_score < img_threshold/100.0) or (curr_txt_score < txt_threshold/100.0):
                continue

            # load images
            org_im = Image.open(invoice_dir + selected_inv)
            dup_im = Image.open(invoice_dir + selected_inv_dups[i])

            left_column, middle_column, right_column = st.columns((4, 1, 4))

            if show_image_diff:
                # read word list for both original and duplicate images
                with open(words_and_boxes_dir + selected_inv + '.wb', 'rb') as fp:
                    w_org, b_org = pickle.load(fp)
                with open(words_and_boxes_dir + selected_inv_dups[i] + '.wb', 'rb') as fp:
                    w_dup, b_dup = pickle.load(fp)
                # find exclusive words idx
                exclusive_org = [w_i for w_i in range(len(w_org)) if w_org[w_i] not in w_dup]
                exclusive_dup = [w_i for w_i in range(len(w_dup)) if w_dup[w_i] not in w_org]


            with left_column:
                st.write('***Original***' + ' ____ Filename: ' + selected_inv)
                if show_image_diff:
                    draw = ImageDraw.Draw(org_im)
                    for w_i in range(len(b_org)):
                        if w_i not in exclusive_org:
                            continue
                        box = b_org[w_i]
                        draw.rectangle(box, outline = 'blue', width = 3)
                st.image(org_im, caption=selected_inv)
                
            with right_column:
                st.write('***Duplicate: ' + str(i) + '/' + str(len(selected_inv_dups)-1) + '***' + ' ____ Filename: ' + selected_inv_dups[i])
                if show_image_diff:
                    draw = ImageDraw.Draw(dup_im)
                    for w_i in range(len(b_dup)):
                        if w_i not in exclusive_dup:
                            continue
                        box = b_dup[w_i]
                        draw.rectangle(box, outline = 'red', width = 3)
                st.image(dup_im, caption=selected_inv_dups[i])
                
            with middle_column:
                st.write("###")
                st.write('')
                st.write("###")
                st.write('')
                st.write('Image Similarity Score:')
                st.write(str(round(curr_img_score*100, 2)))
                st.write('Text Similarity Score:')
                st.write(str(round(curr_txt_score*100, 2)))
                if show_image_diff:
                    total_count = len(w_org) + len(w_dup)
                    diff_count = len(exclusive_org) + len(exclusive_dup)
                    common_count = total_count - diff_count
                    st.write('Word Similarity Score:')
                    st.write(str(round((common_count/total_count)*100, 2)))
                st.write("###")
                st.write('')
                def_choice_index = inv_matching_opts.index(curr_resp)
                ch = st.radio('Choose Option:', inv_matching_opts,
                              key = 'action' + '__' + selected_inv + '__' + selected_inv_dups[i],
                              index = def_choice_index)

                # update duplicates if the response has changed
                if ch != curr_resp:
                    dt.loc[dt.dup == selected_inv_dups[i], "action"] = ch
                    # also mark this pair as seen
                    # dt.loc[dt.dup == selected_inv_dups[i], "seen"] = "seen"
                    # since duplicate is modified, save it again
                    with open(duplicates_dir + selected_inv + '.dup', 'wb') as fp:
                        pickle.dump(dt, fp)
                    # st.experimental_rerun()

                # add a button to go to next/prev invoice
                go_next = st.button("Next Invoice",
                                    key = 'go_next' + '__' + selected_inv + '__' + selected_inv_dups[i])
                go_prev = st.button("Prev Invoice",
                                    key = 'go_prev' + '__' + selected_inv + '__' + selected_inv_dups[i])

                if go_next:
                    # update seen
                    dt.loc[dt.dup == selected_inv_dups[i], "seen"] = "seen"
                    # since duplicate is modified, save it again
                    with open(duplicates_dir + selected_inv + '.dup', 'wb') as fp:
                        pickle.dump(dt, fp)
                    # push current loop iter
                    loop_counter_stack_push(selected_inv, i)
                    st.experimental_rerun()

                if go_prev:
                    # make prev. inv as not_seen. Don't do it for first invoice
                    if i > 0:
                        prev_i = loop_counter_stack_pop(selected_inv)
                        if prev_i is not None:
                            dt.loc[dt.dup == selected_inv_dups[prev_i], "seen"] = "not_seen"
                            # since duplicate is modified, save it again
                            with open(duplicates_dir + selected_inv + '.dup', 'wb') as fp:
                                pickle.dump(dt, fp)
                        st.experimental_rerun()

                # break from loop after one iteration
                break



        st.write('---')
        st.subheader('Duplicate Table')
        with st.container():

            left_column, right_column = st.columns((3,1))

            with left_column:
                show_dup_table = st.checkbox('Show Dup table (invoice: ' + selected_inv + ')', value = True)
                if show_dup_table:
                    df = dt
                    df.reset_index(drop=True, inplace=True)
                    st.dataframe(df)

                    st.download_button(
                        label='Download Dup table for invoice: ' + selected_inv + ' as CSV',
                        data=convert_df_to_csv(df),
                        file_name='output_df' + selected_inv + '.csv',
                        mime='text/csv',
                    )

        with st.container():

            left_column, right_column = st.columns((1,1))

            with left_column:

                st.download_button(
                    label="Download Dup table for all invoices as CSV",
                    data=download_all_image_duplicates(),
                    file_name='output_df.csv',
                    mime='text/csv',
                )







