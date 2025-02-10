# VeriExCite: Verify Existing Citations

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**VeriExCite** is a Python tool designed to help you verify the existence of citations in academic papers (PDF format). It extracts the bibliography section from a PDF, parses individual references, and then checks their validity against Crossref and Google Scholar databases.

## Why VeriExCite?

The rise of powerful LLMs has brought with it the potential for generating realistic-looking, yet entirely fabricated, academic references. While tools like ZeroGPT attempt to detect LLM-generated text, they rely on "black box" deep learning methods, which are prone to both false positives and false negatives, making them unreliable for definitive judgments. However, the presence of fabricated references within a manuscript provides *concrete* evidence that (at least part) of the text may be LLM-generated. VeriExCite focuses on this crucial aspect of academic integrity.

## Features

*   **Extraction:** Extracts the bibliography section from PDF documents.
*   **Parsing:** Uses Google Gemini API to parse references into structured data (title, authors, DOI, type). 
*   **Verification:**
    *   Checks references against Crossref and Google Scholar.
    *   Identifies potentially fabricated citations.
*   **Reporting:**
    *   Provides a summary of verified, skipped (e.g., non-academic references, such as a government website or a dataset), and potentially fabricated references.
    *   Outputs results to a CSV file for easy analysis.
*  **Folder Processing:** Processes all PDF files in a directory in a single run, suitable for academics and teaching assistants in marking scenarios.
*  **Privacy-Conscious:** Only the bibliography section of the PDF is sent to the Google Gemini API. This is crucial for complying with university policies that often prohibit uploading student work (which students hold copyright for) to third-party LLM services. The full text of the paper *is not* uploaded.

## Requirements

1. **Python Libraries:**

   *   `PyPDF2`: For PDF text extraction.
   *   `pydantic`: For data validation and structuring.
   *   `requests`: For making API requests.
   *   `pandas`: For data analysis and CSV output.
   *   `re`: For regular expressions (text processing).
   *   `unidecode`: For handling accented characters.
   *   `scholarly`: For accessing Google Scholar data.
   *   `tenacity`: For implementing retry logic.
   *   `google-genai`: For accessing Google Gemini. 

   **Install requirements:**

   ```
   pip install -r requirements.txt
   ```

   OR

   ```bash
   pip install PyPDF2 pydantic requests pandas unidecode scholarly tenacity google-genai
   ```

3.  **Google Gemini API Key:**
    
    *   Obtain an API key from [Google AI Studio](https://ai.google.dev/aistudio). It's free with 1500 requests per day!
    *   Set your API key to the `GOOGLE_API_KEY` variable in the code. 

## Usage

### 1. Single PDF File

```python
from veriexcite import veriexcite

pdf_path = "path/to/your/paper.pdf"
count_verified, count_warning, count_skipped, list_warning = veci_cite(pdf_path)

print(f"{count_verified} references verified, {count_warning} warnings, {count_skipped} skipped.")

if count_warning > 0:
    print("Warning List:")
    for item in list_warning:
      print(item)
```

### 2. Process a Folder of PDFs

```python
from veriexcite import process_folder

folder_path = "path/to/your/pdf/folder"
process_folder(folder_path)
```
This will create a `VeriCite results.csv` file in the current directory with the results.

## Interpreting Results

*   **Found References:** The total number of references extracted from the bibliography section of the PDF.
*   **Skipped website:** Websites are skipped because they often lack the structured metadata needed for reliable verification.
*   **Verified:** References that were successfully matched in Crossref or Google Scholar.
*   **Warnings:** References that could *not* be verified. 
*   **Warning List:** The raw text of the unverified references. 

> [!IMPORTANT]
>
> A "warning" from VeriExCite indicates suspicion. Manual verification is required. 

