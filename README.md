# invoice-processing

This is a stream-lit based app which creates a tool for invoice processing. Invoices can be loaded in `data/sample_invoices` directory and the app will compute image embedding, word embeddings and check for duplicates. It uses `tesserect` for OCR and `SentenceTransformer` for computing embeddings.
