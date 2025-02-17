import streamlit as st
from veriexcite import (
    extract_bibliography_section,
    split_references,
    search_title,
    set_google_api_key,
)
import io
import pandas as pd
import PyPDF2


def extract_text_from_pdf(pdf_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Validates if the file is a PDF, then extract text."""
    if not pdf_file.name.lower().endswith(".pdf"):
        raise ValueError("Uploaded file is not a PDF.")
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    pdf_content = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pdf_content += page_text + "\n"
    return pdf_content


def process_and_verify(text: str, keywords=["Reference", "Bibliography", "Works Cited"]) -> pd.DataFrame:
    """Extracts, processes, and verifies references."""
    # Create containers in the main area
    progress_text = st.empty()
    placeholder = st.empty()
    progress_text.text("Extracting bibliography ...")

    try:
        bib_text = extract_bibliography_section(text, keywords)
        references = split_references(bib_text)
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    ref_type_dict = {"journal_article": "Journal Article", "conference_paper": "Conference Paper",
                     "book": "Book", "book_chapter": "Book Chapter", "non_academic_website": "Website"}
    status_dict = {"Verified": "✅Verified", "Warning": "⚠️Warning"}

    results = []
    for idx, ref in enumerate(references):
        status = "Pending"
        results.append({
            "Index": idx,
            "First Author": ref.first_author_family_name,
            "Year": str(ref.year),
            "Title": ref.title,
            "Type": ref_type_dict.get(ref.type, ref.type),
            "DOI": ref.DOI,
            "URL": ref.URL,
            "Reference Text": ref.normalised_input_bibliography,
            "Status": status,
        })

    df = pd.DataFrame(results)
    df_display = pd.concat([df[['First Author', 'Year', 'Title', 'Type', 'URL', 'DOI']],
                            df['Status'].map(status_dict)], axis=1)
    placeholder.dataframe(df_display, use_container_width=True,
                          column_config={
                              "URL": st.column_config.LinkColumn()
                          })
    verified_count = 0
    warning_count = 0
    progress_text.text(f"Verified: {verified_count} | Warnings: {warning_count}")

    for index, row in df.iterrows():
        is_verified = search_title(references[index])
        if is_verified:
            df.loc[index, "Status"] = "Verified"
            verified_count += 1
        else:
            df.loc[index, "Status"] = "Warning"
            warning_count += 1

        # add emoji to the status column
        # add a link to the DOI
        df_display = pd.concat([df[['First Author', 'Year', 'Title', 'Type', 'URL', 'DOI']],
                                df['Status'].map(status_dict)], axis=1)
        placeholder.dataframe(df_display, use_container_width=True,
                              column_config={
                                  "URL": st.column_config.LinkColumn()
                              })

        progress_text.text(f"Verified: {verified_count} | Warnings: {warning_count}")

    return df


def main():
    st.set_page_config(page_title="VeriExCite", page_icon="🔍", layout="wide", initial_sidebar_state="expanded",
                       menu_items={
                           "About": "This is a tool to verify citations in academic papers. View the source code on [GitHub](https://github.com/ykangw/VeriExCiting)."})

    st.title("VeriExCite: Verify Existing Citations")
    st.write(
        "This tool helps verify the existence of citations in academic papers (PDF format). "
        "It extracts the bibliography, parses references, and checks their validity."
    )

    with st.sidebar:
        st.header("Input")
        pdf_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

        use_dev_key = st.checkbox("Use developer's API key for a trial (limited uses)")
        st.write(
            "You can apply for a Gemini API key at [Google AI Studio](https://ai.google.dev/aistudio) with 1500 requests per day for FREE.")
        if use_dev_key:
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            api_key = st.text_input("Enter your Google Gemini API key:", type="password")

        if st.button("Start Verification"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file.")
                return

            if not api_key:
                st.warning("Please enter a Google Gemini API key or select 'Use developer's API key'.")
                return


    try:
        set_google_api_key(api_key)
        all_results = []

        for pdf_file in pdf_files:
            subheader = st.subheader(f"Processing: {pdf_file.name}")
            text = extract_text_from_pdf(pdf_file)
            results_df = process_and_verify(text)
            results_df['Source File'] = pdf_file.name
            all_results.append(results_df)
            subheader.subheader(f"Completed: {pdf_file.name}")

        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            csv = combined_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Results as CSV",
                data=csv,
                file_name='VeriCite_results.csv',
                mime='text/csv',
            )

    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()