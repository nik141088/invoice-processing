from PIL import Image, ImageDraw
import streamlit as st
from streamlit_lottie import st_lottie
import os
import pickle
import math

from src.constants import LOTTIE_INFOGRAPHIC_URL, CSS_PATH
from src.constants import INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, WORDS_AND_BOXES_DIR, DUPLICATES_DIR, INV_DONE_DIR, PREV_LOOP_ITERATION_FILE
from src.constants import INVOICE_MATCHING_CHOICES, DEFAULT_INVOICE_MATCHING_CHOICE, IMG_SIMILARITY_CUTOFF, TXT_SIMILARITY_CUTOFF, ABS_TOL

from src.loop_counter import loop_counter_stack_pop, loop_counter_stack_push, loop_counter_stack_clear
from src.style import load_lottie_url, local_css
from src.utilities import create_dirs, initialize_loop_stack, convert_df_to_csv, download_all_image_duplicates, delete_all_files_in_dir
from src.utilities import get_short_names, get_duplicate_files, list_files
from src.core.embeddings import compute_embeddings
from src.core.duplicates import compute_duplicates

# Find more emojis here: /https://www.webfx.com/tools/emoji-cheat-sheet
st.set_page_config(page_title="Invoice-Processing (with Transformers)", page_icon=":muscle:", layout="wide")
# load css and animation
local_css(CSS_PATH)
lottie_invoice = load_lottie_url(LOTTIE_INFOGRAPHIC_URL)

# Set up directories
create_dirs()
initialize_loop_stack()

# ---- header ----
with st.container():
    left_column, right_column = st.columns((1, 1))

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

        # delete data: text/image embeddings, words and boxes, duplicates, invoice dones
        delete_all_files_in_dir(IMAGE_EMBEDDINGS_DIR)
        delete_all_files_in_dir(TEXT_EMBEDDINGS_DIR)
        delete_all_files_in_dir(WORDS_AND_BOXES_DIR)
        delete_all_files_in_dir(DUPLICATES_DIR)
        delete_all_files_in_dir(INV_DONE_DIR)

        # remove PREV_LOOP_ITERATION_FILE
        os.remove(PREV_LOOP_ITERATION_FILE)

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
    delete_all_files_in_dir(INV_DONE_DIR)

    # remove PREV_LOOP_ITERATION_FILE
    os.remove(PREV_LOOP_ITERATION_FILE)

    # display done message
    st.subheader('Model check done!')

if show_duplicates:
    with st.container():
        # recompute duplicate names
        duplicates_names, duplicates_names_short = get_duplicate_files(ret_short_names=True)

        if len(duplicates_names_short) == 0:
            compute_embeddings()
            compute_duplicates()

            # delete all invoice dones
            delete_all_files_in_dir(INV_DONE_DIR)

            # remove PREV_LOOP_ITERATION_FILE
            os.remove(PREV_LOOP_ITERATION_FILE)

            # recompute duplicate_names
            duplicates_names, duplicates_names_short = get_duplicate_files(ret_short_names=True)

        # invoice done files
        done_names = list_files(INV_DONE_DIR)
        done_names_short = get_short_names(done_names, '.done')

        # which invoices out of duplicates_name_shorts are remaining after accounting for done_names_short
        rem_names_short = [d for d in duplicates_names_short if d not in done_names_short]
        if len(rem_names_short) == 0:
            # we are done! stop the app!
            st.subheader('***All invoices done!***')
            st.stop()

        st.write("---")
        st.title("Deep Dive!")

        left_column, middle_column, right_column = st.columns((2, 1, 2))

        with left_column:
            img_threshold = st.slider('Select image similarity cutoff (in %):',
                                      min_value=round(100.0 * IMG_SIMILARITY_CUTOFF, 0),
                                      max_value=100.0,
                                      value=round(0.5 * 100 * IMG_SIMILARITY_CUTOFF + 0.5 * 100.0, 0),
                                      step=1.0)
            txt_threshold = st.slider('Select text similarity cutoff (in %):',
                                      min_value=round(100.0 * TXT_SIMILARITY_CUTOFF, 0),
                                      max_value=100.0,
                                      value=round(0.5 * 100 * TXT_SIMILARITY_CUTOFF + 0.5 * 100.0, 0),
                                      step=1.0)
            # selected_inv = st.selectbox('Select Invoice:', rem_names_short)
            # Alternatively, always choose first element of rem_names_short as selected_inv and remove selectbox
            # we will also need to remove these buttons: redo_selected_invoice, reset_selected_invoice, st.download_button (for invoice level download)
            selected_inv = rem_names_short[0]
            st.write('***Selected Invoice:*** ' + selected_inv)
            skip_actioned_items = st.checkbox("Don't show updated pairs?", value=False)
            skip_seen_items = st.checkbox("Don't show already seen pairs?", value=True)
            hide_pure_duplicates = st.checkbox("Hide pairs with score of 100.0?", value=False)
            show_image_diff = st.checkbox("Highlight text differences?", value=False)

        with middle_column:
            redo_selected_invoice = st.button('Redo selected invoice?')
            if redo_selected_invoice:
                # read duplicate table corresponding to selected invoice
                dup_file = os.path.join(DUPLICATES_DIR, selected_inv + '.dup')
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
                dup_file = os.path.join(DUPLICATES_DIR, selected_inv + '.dup')
                with open(dup_file, 'rb') as fp:
                    dt = pickle.load(fp)
                # set seen to unseen
                dt["action"] = DEFAULT_INVOICE_MATCHING_CHOICE
                dt["seen"] = "not_seen"
                # save it
                with open(dup_file, 'wb') as fp:
                    pickle.dump(dt, fp)
                # push current loop iter
                loop_counter_stack_clear()

        # read duplicate table corresponding to selected invoice
        with open(os.path.join(DUPLICATES_DIR, selected_inv + '.dup'), 'rb') as fp:
            dt = pickle.load(fp)
        dt = dt.sort_values(["filecmp", "img_score", "txt_score"], ascending=[False, False, False])

        # list of possible duplicate invoices
        filter_dt = dt[(dt.img_score >= img_threshold / 100.0) & (dt.txt_score >= txt_threshold / 100.0)]
        if skip_actioned_items:
            filter_dt = filter_dt[(filter_dt.action == DEFAULT_INVOICE_MATCHING_CHOICE)]
        if skip_seen_items:
            filter_dt = filter_dt[(filter_dt.seen == "not_seen")]
        if hide_pure_duplicates:
            filter_dt = filter_dt[(filter_dt.filecmp == False)]
            filter_dt = filter_dt[(filter_dt.img_score <= 1.0 - ABS_TOL) | (filter_dt.txt_score <= 1.0 - ABS_TOL)]

        # choose dup invoice if there is at least one entry present
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
                    with open(os.path.join(INV_DONE_DIR, selected_inv + '.done'), 'wb') as fp:
                        pickle.dump(None, fp)
                    # clear loop iteration stack
                    loop_counter_stack_clear()
                    # rerun
                    st.experimental_rerun()

        for i in range(len(selected_inv_dups)):
            curr_df = dt[dt.dup == selected_inv_dups[i]]

            curr_resp = curr_df.action.to_numpy()[0]
            curr_seen = curr_df.seen.to_numpy()[0]
            curr_filecmp = curr_df.filecmp.to_numpy()[0]
            curr_img_score = curr_df.img_score.to_numpy()[0]
            curr_txt_score = curr_df.txt_score.to_numpy()[0]

            if skip_seen_items and curr_seen == "seen":
                continue

            if skip_actioned_items and curr_resp != DEFAULT_INVOICE_MATCHING_CHOICE:
                continue

            if hide_pure_duplicates and ((curr_filecmp is True) or
                                         (math.isclose(curr_img_score, 1.0, abs_tol=ABS_TOL) and
                                          math.isclose(curr_txt_score, 1.0, abs_tol=ABS_TOL))):
                continue

            if (curr_img_score < img_threshold / 100.0) or (curr_txt_score < txt_threshold / 100.0):
                continue

            # load images
            org_im = Image.open(os.path.join(INVOICE_DIR, selected_inv))
            dup_im = Image.open(os.path.join(INVOICE_DIR, selected_inv_dups[i]))

            left_column, middle_column, right_column = st.columns((4, 1, 4))

            if show_image_diff:
                # read word list for both original and duplicate images
                with open(os.path.join(WORDS_AND_BOXES_DIR, selected_inv + '.wb'), 'rb') as fp:
                    w_org, b_org = pickle.load(fp)
                with open(os.path.join(WORDS_AND_BOXES_DIR, selected_inv_dups[i] + '.wb'), 'rb') as fp:
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
                        draw.rectangle(box, outline='blue', width=3)
                st.image(org_im, caption=selected_inv)

            with right_column:
                st.write('***Duplicate: ' + str(i) + '/' + str(len(selected_inv_dups) - 1) + '***' + ' ____ Filename: ' + selected_inv_dups[i])
                if show_image_diff:
                    draw = ImageDraw.Draw(dup_im)
                    for w_i in range(len(b_dup)):
                        if w_i not in exclusive_dup:
                            continue
                        box = b_dup[w_i]
                        draw.rectangle(box, outline='red', width=3)
                st.image(dup_im, caption=selected_inv_dups[i])

            with middle_column:
                st.write("###")
                st.write('')
                st.write("###")
                st.write('')
                st.write('Image Similarity Score:')
                st.write(str(round(curr_img_score * 100, 2)))
                st.write('Text Similarity Score:')
                st.write(str(round(curr_txt_score * 100, 2)))
                if show_image_diff:
                    total_count = len(w_org) + len(w_dup)
                    diff_count = len(exclusive_org) + len(exclusive_dup)
                    common_count = total_count - diff_count
                    st.write('Word Similarity Score:')
                    st.write(str(round((common_count / total_count) * 100, 2)))
                st.write("###")
                st.write('')
                def_choice_index = INVOICE_MATCHING_CHOICES.index(curr_resp)
                ch = st.radio('Choose Option:', INVOICE_MATCHING_CHOICES,
                              key='action' + '__' + selected_inv + '__' + selected_inv_dups[i],
                              index=def_choice_index)

                # update duplicates if the response has changed
                if ch != curr_resp:
                    dt.loc[dt.dup == selected_inv_dups[i], "action"] = ch
                    # also mark this pair as seen
                    # dt.loc[dt.dup == selected_inv_dups[i], "seen"] = "seen"
                    # since duplicate is modified, save it again
                    with open(os.path.join(DUPLICATES_DIR, selected_inv + '.dup'), 'wb') as fp:
                        pickle.dump(dt, fp)
                    # st.experimental_rerun()

                # add a button to go to next/prev invoice
                go_next = st.button("Next Invoice",
                                    key='go_next' + '__' + selected_inv + '__' + selected_inv_dups[i])
                go_prev = st.button("Prev Invoice",
                                    key='go_prev' + '__' + selected_inv + '__' + selected_inv_dups[i])

                if go_next:
                    # update seen
                    dt.loc[dt.dup == selected_inv_dups[i], "seen"] = "seen"
                    # since duplicate is modified, save it again
                    with open(os.path.join(DUPLICATES_DIR, selected_inv + '.dup'), 'wb') as fp:
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
                            with open(os.path.join(DUPLICATES_DIR, selected_inv + '.dup'), 'wb') as fp:
                                pickle.dump(dt, fp)
                        st.experimental_rerun()

                # break from loop after one iteration
                break

        st.write('---')
        st.subheader('Duplicate Table')
        with st.container():

            left_column, right_column = st.columns((3, 1))

            with left_column:
                show_dup_table = st.checkbox('Show Dup table (invoice: ' + selected_inv + ')', value=True)
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

            left_column, right_column = st.columns((1, 1))

            with left_column:
                st.download_button(
                    label="Download Dup table for all invoices as CSV",
                    data=download_all_image_duplicates(),
                    file_name='output_df.csv',
                    mime='text/csv',
                )
