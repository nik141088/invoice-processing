from PIL import Image, ImageDraw
import streamlit as st
from streamlit_lottie import st_lottie
import os
import pickle

from src.constants import (
    LOTTIE_INFOGRAPHIC_URL, CSS_PATH, INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR,
    WORDS_AND_BOXES_DIR, DUPLICATES_DIR, INV_DONE_DIR, PREV_LOOP_ITERATION_FILE, DEFAULT_INVOICE_MATCHING_CHOICE,
    IMG_SIMILARITY_CUTOFF, TXT_SIMILARITY_CUTOFF
)
from src.loop_counter import loop_counter_stack_clear
from src.style import load_lottie_url, local_css
from src.utilities import create_dirs, initialize_loop_stack, delete_all_files_in_dir, get_short_names, get_duplicate_files, list_files
from src.core.embeddings import compute_embeddings
from src.core.duplicates import compute_duplicates


def setup_page():
    """Set up the Streamlit page."""
    st.set_page_config(
        page_title="Invoice-Processing (with Transformers)",
        page_icon=":muscle:",
        layout="wide"
    )
    local_css(CSS_PATH)


def show_header():
    """Display the app header."""
    with st.container():
        left_column, right_column = st.columns((1, 1))
        with left_column:
            st.title(":muscle: Invoice-Processing (with Transformers)")
            st.subheader("Process Duplicate Invoices!")
            st.write(
                "This app lets you detect and process duplicate invoices! It also handles rotated images!"
            )
            st.write("##")
            st.subheader("Let's Go!")
        with right_column:
            lottie_animation = load_lottie_url(LOTTIE_INFOGRAPHIC_URL)
            st_lottie(lottie_animation, height=400, key="invoice")


def handle_refresh_db():
    """Handle database refresh logic."""
    if st.checkbox("Refresh Database?"):
        if st.checkbox("Confirm? Everything will be computed afresh!"):
            delete_all_files_in_dir(IMAGE_EMBEDDINGS_DIR)
            delete_all_files_in_dir(TEXT_EMBEDDINGS_DIR)
            delete_all_files_in_dir(WORDS_AND_BOXES_DIR)
            delete_all_files_in_dir(DUPLICATES_DIR)
            delete_all_files_in_dir(INV_DONE_DIR)
            os.remove(PREV_LOOP_ITERATION_FILE)
            compute_embeddings()
            compute_duplicates()
            st.subheader('Model computation done!')


def handle_data_change():
    """Handle logic when data changes."""
    if st.checkbox("Data has changed?"):
        compute_embeddings()
        compute_duplicates()
        delete_all_files_in_dir(INV_DONE_DIR)
        os.remove(PREV_LOOP_ITERATION_FILE)
        st.subheader('Model check done!')


def display_duplicates():
    """Display duplicate processing workflow."""
    if not st.checkbox("Show Duplicates"):
        return

    duplicates_names, duplicates_names_short = get_duplicate_files(ret_short_names=True)

    if len(duplicates_names_short) == 0:
        compute_embeddings()
        compute_duplicates()
        duplicates_names, duplicates_names_short = get_duplicate_files(ret_short_names=True)

    remaining_names_short = get_remaining_duplicates(duplicates_names_short)
    if not remaining_names_short:
        st.subheader('***All invoices done!***')
        st.stop()

    process_invoice_duplicates(remaining_names_short[0])


def get_remaining_duplicates(duplicates, done_files_suffix=".done"):
    """Get remaining duplicates by excluding done files."""
    done_files = list_files(INV_DONE_DIR)
    done_files_short = get_short_names(done_files, done_files_suffix)
    return [d for d in duplicates if d not in done_files_short]


def process_invoice_duplicates(selected_invoice):
    """Process duplicates for a selected invoice."""
    dt = load_duplicate_invoice(selected_invoice)
    dt = filter_duplicates(dt)
    selected_duplicates = get_selected_duplicates(dt)

    if not any(selected_duplicates):
        mark_invoice_done(selected_invoice)
        return

    show_duplicate_comparison(dt, selected_invoice, selected_duplicates)


def load_duplicate_invoice(selected_invoice):
    """Load a duplicate invoice from disk."""
    with open(os.path.join(DUPLICATES_DIR, selected_invoice + '.dup'), 'rb') as fp:
        return pickle.load(fp)


def filter_duplicates(df):
    """Filter duplicates based on user settings."""
    img_threshold = st.slider('Select image similarity cutoff (in %):', min_value=round(100.0*IMG_SIMILARITY_CUTOFF, 0), max_value=100.0, value=round(0.5*100*IMG_SIMILARITY_CUTOFF + 0.5*100.0, 0), step=1.0)
    txt_threshold = st.slider('Select text similarity cutoff (in %):', min_value=round(100.0*TXT_SIMILARITY_CUTOFF, 0), max_value=100.0, value=round(0.5*100*TXT_SIMILARITY_CUTOFF + 0.5*100.0, 0), step=1.0)
    skip_actioned_items = st.checkbox("Skip actioned items?", value=False)
    skip_seen_items = st.checkbox("Skip seen items?", value=True)
    hide_pure_duplicates = st.checkbox("Hide perfect matches?", value=False)

    filtered_df = df[(df.img_score >= img_threshold / 100) & (df.txt_score >= txt_threshold / 100)]
    if skip_actioned_items:
        filtered_df = filtered_df[filtered_df.action == DEFAULT_INVOICE_MATCHING_CHOICE]
    if skip_seen_items:
        filtered_df = filtered_df[filtered_df.seen == "not_seen"]
    if hide_pure_duplicates:
        filtered_df = filtered_df[(df.filecmp == False)]
    return filtered_df


def get_selected_duplicates(df):
    """Get the list of duplicates to process."""
    return df.dup.to_numpy() if not df.empty else []


def mark_invoice_done(selected_invoice):
    """Mark an invoice as done."""
    with open(os.path.join(INV_DONE_DIR, selected_invoice + '.done'), 'wb') as fp:
        pickle.dump(None, fp)
    loop_counter_stack_clear()
    st.experimental_rerun()


def show_duplicate_comparison(df, selected_invoice, selected_duplicates):
    """Show side-by-side comparison of duplicates."""
    show_image_diff = st.checkbox("Highlight text differences?", value=False)
    for i, duplicate in enumerate(selected_duplicates):
        display_comparison(selected_invoice, duplicate, df, show_image_diff)
        handle_navigation_buttons(selected_invoice, i, selected_duplicates)


def highlight_differences(selected_inv, duplicate, show_image_diff):
    """
    Highlight differences between original and duplicate invoices.
    """
    if not show_image_diff:
        return None, None

    # Load word boxes and words for both original and duplicate images
    with open(os.path.join(WORDS_AND_BOXES_DIR, selected_inv + '.wb'), 'rb') as fp:
        w_org, b_org = pickle.load(fp)

    with open(os.path.join(WORDS_AND_BOXES_DIR, duplicate + '.wb'), 'rb') as fp:
        w_dup, b_dup = pickle.load(fp)

    # Find exclusive words in original and duplicate
    exclusive_org = [idx for idx, word in enumerate(w_org) if word not in w_dup]
    exclusive_dup = [idx for idx, word in enumerate(w_dup) if word not in w_org]

    # Draw bounding boxes for exclusive words
    org_im = Image.open(os.path.join(INVOICE_DIR, selected_inv))
    dup_im = Image.open(os.path.join(INVOICE_DIR, duplicate))
    draw_org = ImageDraw.Draw(org_im)
    draw_dup = ImageDraw.Draw(dup_im)

    for idx in exclusive_org:
        box = b_org[idx]
        draw_org.rectangle(box, outline="blue", width=3)

    for idx in exclusive_dup:
        box = b_dup[idx]
        draw_dup.rectangle(box, outline="red", width=3)

    return org_im, dup_im


def display_comparison(selected_invoice, duplicate, df, show_image_diff):
    """
    Display a side-by-side comparison of original and duplicate invoices with optional differences highlighted.
    """
    left_col, middle_col, right_col = st.columns((4, 1, 4))

    org_im, dup_im = highlight_differences(selected_invoice, duplicate, show_image_diff)

    with left_col:
        st.write(f"***Original Invoice***: {selected_invoice}")
        if org_im:
            st.image(org_im, caption=selected_invoice)
        else:
            display_image(selected_invoice)

    with right_col:
        st.write(f"***Duplicate Invoice***: {duplicate}")
        if dup_im:
            st.image(dup_im, caption=duplicate)
        else:
            display_image(duplicate)

    with middle_col:
        show_comparison_metadata(df, selected_invoice, duplicate)


def display_image(invoice):
    """Display an invoice image."""
    img_path = os.path.join(INVOICE_DIR, invoice)
    st.image(Image.open(img_path), caption=invoice)


def show_comparison_metadata(df, original, duplicate):
    """Show metadata for the current comparison."""
    record = df[df.dup == duplicate].iloc[0]
    st.write("Image Similarity:", round(record.img_score * 100, 2), "%")
    st.write("Text Similarity:", round(record.txt_score * 100, 2), "%")


def handle_navigation_buttons(selected_invoice, index, selected_duplicates):
    """Handle navigation buttons for next/previous duplicates."""
    go_next = st.button("Next", key=f"next_{index}")
    go_prev = st.button("Previous", key=f"prev_{index}")

    if go_next:
        update_seen_status(selected_invoice, selected_duplicates[index], seen=True)
        st.experimental_rerun()
    if go_prev:
        update_seen_status(selected_invoice, selected_duplicates[index - 1], seen=False)
        st.experimental_rerun()


def update_seen_status(selected_invoice, duplicate, seen=True):
    """Update the 'seen' status of a duplicate."""
    with open(os.path.join(DUPLICATES_DIR, selected_invoice + '.dup'), 'rb') as fp:
        df = pickle.load(fp)
    df.loc[df.dup == duplicate, "seen"] = "seen" if seen else "not_seen"
    with open(os.path.join(DUPLICATES_DIR, selected_invoice + '.dup'), 'wb') as fp:
        pickle.dump(df, fp)


def main():
    """Main function to run the Streamlit app."""
    setup_page()
    show_header()
    create_dirs()
    initialize_loop_stack()
    handle_refresh_db()
    handle_data_change()
    display_duplicates()


if __name__ == "__main__":
    main()
