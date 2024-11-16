import os
import win32file


# pytesseract path/config (on my computer)
PYTESSERACT_EXECUTABLE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-21.03.0\Library\bin"
TESSDATA_DIR_CONFIG = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# style related / lottie
LOTTIE_INFOGRAPHIC_URL = "https://assets2.lottiefiles.com/packages/lf20_g07hYi.json"
CSS_PATH = "app/style/style.css"

# Data directories
ROOT_DIR = 'data'
# all the invoices are present in invoice_dir.
# We use img_embeddings_dir to create image imbeddings for all the images
# all the duplicates are saved in duplicates_dir. Duplicates are saved wrt each invoice.
# inv_done_dir is used to cache invoices as and when all their duplicates are updated. This is required so that one can pick up dup checking from where they left
INVOICE_DIR = os.path.join(ROOT_DIR, 'sample_invoices')
IMAGE_EMBEDDINGS_DIR = os.path.join(ROOT_DIR, 'img_emb')
TEXT_EMBEDDINGS_DIR = os.path.join(ROOT_DIR, 'txt_emb')
WORDS_AND_BOXES_DIR = os.path.join(ROOT_DIR, 'words_and_boxes')
DUPLICATES_DIR = os.path.join(ROOT_DIR, 'duplicates')
INV_DONE_DIR = os.path.join(ROOT_DIR, 'invDone')
PREV_LOOP_ITERATION_FILE = os.path.join(ROOT_DIR, 'prev_i')

# Possible options for invoice matching
INVOICE_MATCHING_CHOICES = ["No Action yet!", "Different", "Pure Duplicate", "Near Duplicate", "More Action"]
DEFAULT_INVOICE_MATCHING_CHOICE = INVOICE_MATCHING_CHOICES[0]

# Some constants governing the default cutoffs for similarity score cutoffs
# it's very unlikely that two copies of the same invoice will have a similarity less than this
IMG_SIMILARITY_CUTOFF = 0.9
TXT_SIMILARITY_CUTOFF = 0.7
ABS_TOL = 1e-5

# Max number of files that can be opened at once (in windows)
DEFAULT_WIN32_MAX_FILE_OPEN = win32file._getmaxstdio()
