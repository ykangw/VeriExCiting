import PyPDF2
from pydantic import BaseModel
import requests
import io
import os
import pandas as pd
import re
from unidecode import unidecode
from scholarly import scholarly
# from scholarly import ProxyGenerator
import logging
from typing import List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai.types import Tool, GoogleSearch
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from enum import Enum

# TODO:
#  Check for API rate limits and explore possible optimizations to accelerate the process
#  Support Chinese reference search, like Baidu Scholar or CNKI https://github.com/1049451037/MagicBaiduScholar

# --- Configuration ---
GOOGLE_API_KEY = None
OPENALEX_BASE_URL = "https://api.openalex.org/works"
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO")
# Unset by default: OpenAlex rejects the retired data-version=1 with a 400, and omitting the
# parameter tracks whatever version is current. Set it only to pin a specific version.
OPENALEX_DATA_VERSION = os.getenv("OPENALEX_DATA_VERSION")
LOBID_BASE_URL = "https://lobid.org/resources/search"

# Default Gemini models per role, override with GEMINI_PARSE_MODEL / GEMINI_SEARCH_MODEL
DEFAULT_MODELS = {
    "PARSE": "gemini-3.5-flash-lite",
    "SEARCH": "gemini-2.5-flash",
}

MAX_URL_DOWNLOAD_BYTES = 10 * 1024 * 1024  # Covers ordinary reports; stops a huge PDF from filling memory.

DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# TODO: Proxy support for scholarly
#  This does not work because scholarly proxy needs https<0.28.0 but gemini requires https>=0.28.0
# # Set up a ProxyGenerator object to use free proxies for Google Scholar API
# # This needs to be done only once per session
# pg = ProxyGenerator()
# pg.FreeProxies()
# scholarly.use_proxy(pg)

def set_google_api_key(api_key: str):
    """Set Google Gemini API key."""
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key


def generate_content(role: str, contents, config):
    """Call Gemini with the model configured for the given role (see DEFAULT_MODELS)."""
    model = os.getenv(f"GEMINI_{role}_MODEL") or DEFAULT_MODELS[role]
    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client.models.generate_content(model=model, contents=contents, config=config)


def grounded_search(prompt: str):
    """Run a Google Search grounded lookup, or return None if it could not be run.

    A quota error or any other Gemini failure must not abort the whole batch: the caller skips
    this source and reports UNCHECKED, so the remaining references are still verified.
    Caught inside the call so the @retry decorators do not spend backoff on a dead quota.
    """
    try:
        return generate_content(
            "SEARCH",
            contents=prompt,
            config={
                'tools': [Tool(google_search=GoogleSearch())],
                'temperature': 0,
            },
        )
    except Exception as e:
        logging.warning(f"Google Search unavailable, skipping this source: {e}")
        return None


def answers_true(response) -> bool:
    """True if a grounded yes/no response affirms the reference exists.
    A grounded call that finds nothing comes back with finish_reason STOP but no parts at all,
    so an empty response must read as 'not found' rather than raise.
    """
    answer = normalize_title(response.text or "")
    return answer.startswith('true') or answer.endswith('true')


# --- Step 1: Read PDF and extract bibliography section ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of the PDF."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_bibliography_section(text: str, keywords: List[str] = [
    # English
    "Reference", "References", "Bibliography", "Works Cited",
    # Chinese
    "参考文献", "參考文獻",
    # Japanese
    "参考資料",
    # French
    "Références", "Bibliographie",
    # German
    "Literaturverzeichnis", "Quellenverzeichnis",
    # Spanish
    "Referencias", "Bibliografía",
    # Russian
    "Список литературы",
    # Italian
    "Riferimenti", "Bibliografia",
    # Portuguese
    "Referências", "Bibliografia",
    # Korean
    "참고문헌"
]) -> str:
    """
    Find the last occurrence of any keyword from 'keywords'
    and return the text from that point onward.
    """
    last_index = -1
    for keyword in keywords:
        index = text.lower().rfind(keyword.lower())
        if index > last_index:
            last_index = index
    if last_index == -1:
        raise ValueError("No bibliography section found using keywords: " + ", ".join(keywords))
    return text[last_index:]


# --- Step 2: Split the bibliography text into individual references ---
class ReferenceExtraction(BaseModel):
    title: str
    author: str
    DOI: str
    URL: str
    year: int
    type: str
    bib: str

class ReferenceStatus(Enum):
    VALIDATED = "validated"
    INVALID = "invalid"
    NOT_FOUND = "not_found"
    # A check could not be run at all (e.g. the Gemini quota is exhausted). Distinct from
    # NOT_FOUND on purpose: no answer is not evidence that a reference was fabricated.
    UNCHECKED = "unchecked"

class ReferenceCheckResult(BaseModel):
    status: ReferenceStatus
    explanation: str


def search_unavailable() -> ReferenceCheckResult:
    """Result used when a Google Search backed check had to be skipped."""
    return ReferenceCheckResult(
        status=ReferenceStatus.UNCHECKED,
        explanation="Google Search unavailable (Gemini quota or API error); this source was skipped.",
    )

def split_references(bib_text):
    """Splits the bibliography text into individual references using the Google Gemini API."""

    prompt = """
    Process a reference list extracted from a PDF, where formatting may be corrupted.  
    Follow these steps to clean and extract key information: 
    1. Normalisation: Fix spacing errors, line breaks, and punctuation.
    2. Extraction: For each reference, extract:
    - Title (full title case)
    - Author: First author's family name (If the author is an organization, use the organization name)
    - DOI (include if explicitly stated; otherwise leave blank)
    - URL (include if explicitly stated; otherwise leave blank)
    - Year (4-digit publication year)
    - Type (journal_article, preprint, conference_paper, book, book_chapter, OR non_academic_website. If the author is not a human but an organization, select non_academic_website)
    - Bib: Normalised input bibliography (correct format, in one line)\n\n
    """

    response = generate_content(
        "PARSE",
        contents=prompt + bib_text,
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[ReferenceExtraction],
            'temperature': 0,
            # No thinking_config: the Gemini 3.x family rejects thinking_budget=0 with a
            # 400 INVALID_ARGUMENT, and its default (dynamic) budget is fine for this task.
        },
    )

    # print(response.text)  # JSON string.
    references: list[ReferenceExtraction] = response.parsed  # Parsed JSON.
    return references


# --- Step 3: Verify each reference using crossref and compare title ---
def normalize_title(title: str) -> str:
    """Normalizes a title for comparison (case-insensitive, no punctuation, etc.)."""
    title = unidecode(title)  # Remove accents
    title = re.sub(r'[^\w\s]', '', title).lower()  # Remove punctuation
    title = re.sub(r'\band\b|\bthe\b', '', title)  # Remove 'and' and 'the'
    title = re.sub(r'\s+', '', title).strip()  # Remove extra whitespace
    return title


TITLE_FUZZ_THRESHOLD = 85


def classify_title_match(candidate: str, reference: str) -> Optional[str]:
    """Returns 'exact', 'partial', 'fuzzy', or None for two normalize_title() outputs.

    This is the comparison every source in this module makes; keeping it in one place means the
    threshold and the containment rule are tuned once. Callers pass already-normalized strings so
    a loop over candidate records can hoist the reference title's normalization out.
    """
    if not candidate or not reference:
        return None
    if candidate == reference:
        return "exact"
    if reference in candidate or candidate in reference:
        return "partial"
    if fuzz.ratio(candidate, reference) > TITLE_FUZZ_THRESHOLD:
        return "fuzzy"
    return None


def normalize_author_name(author: str) -> str:
    """Returns a lowercase surname/organization token for comparison."""
    if not author:
        return ""
    normalized = unidecode(author).lower()
    has_comma = "," in normalized
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    parts = normalized.split()
    if not parts:
        return ""
    if has_comma:
        return parts[0]
    return parts[-1]


def _extract_author_search_token(author: str) -> str:
    """Returns a surname token with original casing for catalog queries."""
    if not author:
        return ""
    author = author.strip()
    if not author:
        return ""
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_scholarly(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using scholarly, with error handling and retries."""
    try:
        search_results = scholarly.search_pubs(ref.title)
        result = next(search_results, None)  # Safely get the first result, or None
        normalized_input_title = normalize_title(ref.title)

        # Check if the first author's family name and title match
        if result and 'bib' in result and 'author' in result['bib'] and 'title' in result['bib']:
            if result['bib']['author'][0].split()[-1] == ref.author:
                match_type = classify_title_match(normalize_title(result['bib']['title']), normalized_input_title)
                if match_type:
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED,
                                                explanation=f"Author and title match Google Scholar ({match_type} match).")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Google Scholar.")
    except Exception as e:
        message = str(e)
        if "Cannot Fetch from Google Scholar" in message:
            logging.info(f"Google Scholar blocked automated query for title '{ref.title}'.")
            explanation = "Google Scholar blocked automated access. Please verify manually or try again later."
        else:
            logging.warning(f"Scholarly search failed for title '{ref.title}': {e}")
            explanation = f"Google Scholar search failed: {e}"
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=explanation)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_openalex(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches OpenAlex for the reference."""
    try:
        title_query = re.sub(r"[,|]", " ", ref.title).strip()
        params = {
            "per-page": 5,
            "filter": f"title.search:{title_query}",
        }
        if OPENALEX_DATA_VERSION:
            params["data-version"] = OPENALEX_DATA_VERSION
        # Deliberately no publication-date filter: OpenAlex often records a different year than
        # the citation (e.g. a later reissue), which would drop otherwise valid matches. The
        # title and author comparison below is what establishes the match.
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO

        headers = {"User-Agent": "VeriExCite/0.1.0"}
        response = requests.get(OPENALEX_BASE_URL, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            logging.warning(f"OpenAlex request failed with status code {response.status_code}")
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND,
                                        explanation=f"OpenAlex request failed with status code {response.status_code}.")

        data = response.json()
        results = data.get("results", [])
        if not results:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in OpenAlex.")

        normalized_input_title = normalize_title(ref.title)
        normalized_ref_author = normalize_author_name(ref.author)
        potential_invalid_result = None

        def _normalize_doi(doi: str) -> str:
            if not doi:
                return ""
            doi = doi.strip()
            lowered = doi.lower()
            for prefix in ("https://doi.org/", "http://doi.org/"):
                if lowered.startswith(prefix):
                    return doi[len(prefix):]
            if lowered.startswith("doi:"):
                return doi[4:]
            return doi

        for item in results:
            item_title = item.get("display_name")
            if not item_title:
                continue
            match_type = classify_title_match(normalize_title(item_title), normalized_input_title)

            # Prefer DOI match when available
            item_doi = item.get("ids", {}).get("doi")
            if ref.DOI and item_doi:
                if _normalize_doi(ref.DOI) == _normalize_doi(item_doi):
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="DOI matches OpenAlex record.")

            if not match_type:
                continue

            author_match = False
            if normalized_ref_author:
                for authorship in item.get("authorships", []):
                    candidate_name = authorship.get("author", {}).get("display_name")
                    if normalize_author_name(candidate_name) == normalized_ref_author:
                        author_match = True
                        break

            if author_match or not normalized_ref_author:
                explanation = f"Author and title match OpenAlex record ({match_type} title match)."
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=explanation)

            if not potential_invalid_result:
                potential_invalid_result = ReferenceCheckResult(
                    status=ReferenceStatus.INVALID,
                    explanation=f"Title matches OpenAlex record ({match_type}) but author does not match."
                )

        if potential_invalid_result:
            return potential_invalid_result
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in OpenAlex.")
    except Exception as e:
        logging.warning(f"OpenAlex search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"OpenAlex search failed: {e}")


def _extract_year_from_publication(publication_nodes: List[dict]) -> str:
    """Extracts a 4-digit publication year from lobid publication entries."""
    for publication in publication_nodes or []:
        for key in ("startDate", "dateStatement"):
            value = publication.get(key)
            if not value:
                continue
            match = re.search(r"\d{4}", value)
            if match:
                return match.group(0)
    return ""


def _escape_lucene_term(term: str, preserve_wildcards: bool = False) -> str:
    """Escapes Lucene special characters unless explicit wildcards should be preserved."""
    specials = set(r'+-&&||!(){}[]^"~*?:\/')
    escaped = []
    for char in term:
        if char in specials and not (preserve_wildcards and char in {"*", "?"}):
            escaped.append(f"\\{char}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _field_word_clauses(text: str, field: str, limit: int = 3) -> List[str]:
    """Creates field-specific clauses for up to `limit` meaningful tokens."""
    if not text:
        return []
    tokens = []
    if isinstance(text, list):
        for part in text:
            if isinstance(part, dict):
                tokens.extend(re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", part.get("label", "")))
            else:
                tokens.extend(re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", str(part)))
    else:
        tokens = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    words = tokens
    if not words:
        return []
    keywords = [w for w in words if len(w) > 3] or words
    clauses = []
    for keyword in keywords[:limit]:
        clauses.append(f"{field}:{_escape_lucene_term(keyword)}")
    return clauses


def _split_title_and_subtitle(title: str) -> Tuple[str, str]:
    """Attempts to split a reference title into main title and subtitle text."""
    if not title:
        return "", ""
    separators = [":", " - ", " – ", " — ", ". "]
    for sep in separators:
        if sep in title:
            head, tail = title.split(sep, 1)
            return head.strip(), tail.strip()
    words = title.split()
    if len(words) > 6:
        head = " ".join(words[:4])
        tail = " ".join(words[4:])
        return head.strip(), tail.strip()
    return title.strip(), ""


def _build_lobid_title_query(title: str) -> str:
    """Builds a resilient title query targeting main and subtitle fields."""
    if not title:
        return ""
    main_part, subtitle_part = _split_title_and_subtitle(title)
    clauses = _field_word_clauses(main_part, "title", limit=4)
    if subtitle_part:
        clauses.extend(_field_word_clauses(subtitle_part, "otherTitleInformation", limit=4))
    if not clauses:
        clauses = _field_word_clauses(title, "title", limit=3)
    return " AND ".join(dict.fromkeys(clauses))


def _build_author_query(author: str) -> List[str]:
    """Returns a list of increasingly relaxed author clauses."""
    clauses = []
    if not author:
        return clauses

    # Exact label match (e.g., "Smith, Scott")
    exact = author.strip()
    if exact:
        clauses.append(f'contribution.agent.label:"{_escape_lucene_term(exact)}"')

    # Bare surname match (e.g., Smith)
    surname_token = _extract_author_search_token(author)
    if surname_token:
        clauses.append(f'contribution.agent.label:{_escape_lucene_term(surname_token)}')

    # Normalized wildcard fallback (e.g., *smith*)
    normalized = normalize_author_name(author)
    if normalized:
        clauses.append(f'contribution.agent.label:*{_escape_lucene_term(normalized)}*')

    return clauses


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_lobid(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches the hbz (lobid) catalog for matching records."""
    try:
        # Use field queries documented at https://lobid.org/resources/api to combine title and contributor filters.
        title_query = _build_lobid_title_query(ref.title)
        if not title_query:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Invalid title for hbz search.")
        query_parts = [f"({title_query})"]
        author_clauses = _build_author_query(ref.author)
        query = " AND ".join(query_parts + author_clauses[:1]) if author_clauses else " AND ".join(query_parts)
        params = {
            "q": query,
            "size": 5,
            "format": "json",
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "VeriExCite/0.1.0",
        }
        response = requests.get(LOBID_BASE_URL, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            logging.warning(f"hbz request failed with status code {response.status_code}")
            return ReferenceCheckResult(
                status=ReferenceStatus.NOT_FOUND,
                explanation=f"hbz request failed with status code {response.status_code}.",
            )

        data = response.json()
        members = data.get("member", [])
        if not members:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in hbz.")

        normalized_input_title = normalize_title(ref.title)
        normalized_ref_author = normalize_author_name(ref.author)
        potential_invalid_result = None

        for item in members:
            item_title = item.get("title")
            if not item_title:
                continue
            match_type = classify_title_match(normalize_title(item_title), normalized_input_title)
            if not match_type:
                continue

            author_match = False
            if normalized_ref_author:
                for contribution in item.get("contribution", []):
                    agent = contribution.get("agent") or {}
                    candidate_name = agent.get("label")
                    if candidate_name and normalize_author_name(candidate_name) == normalized_ref_author:
                        author_match = True
                        break
            else:
                author_match = True

            publication_year = _extract_year_from_publication(item.get("publication", []))

            if author_match:
                explanation = f"Title found in hbz catalog ({match_type} title match)."
                if publication_year:
                    explanation += f" Publication year {publication_year}."
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation=explanation)

            if not potential_invalid_result:
                invalid_explanation = f"Title matches hbz catalog ({match_type}) but author does not match."
                if publication_year:
                    invalid_explanation += f" Catalog year {publication_year}."
                potential_invalid_result = ReferenceCheckResult(
                    status=ReferenceStatus.INVALID,
                    explanation=invalid_explanation,
                )

        if potential_invalid_result:
            return potential_invalid_result
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in hbz.")
    except Exception as e:
        logging.warning(f"hbz search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"hbz search failed: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_doi_crossref(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a DOI using the Crossref API, with retries. Returns ReferenceCheckResult."""
    if not ref.DOI:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No DOI provided for DOI-based search.")
    
    try:
        # Clean DOI by removing potential URL prefix
        clean_doi = ref.DOI.strip()
        if clean_doi.startswith('https://doi.org/'):
            clean_doi = clean_doi[16:]  # Remove 'https://doi.org/'
        elif clean_doi.startswith('http://doi.org/'):
            clean_doi = clean_doi[15:]  # Remove 'http://doi.org/'
        elif clean_doi.startswith('doi:'):
            clean_doi = clean_doi[4:]  # Remove 'doi:' prefix
        
        # Search by DOI directly
        response = requests.get(f"https://api.crossref.org/works/{clean_doi}")
        
        if response.status_code == 200:
            item = response.json().get('message', {})
            normalized_input_title = normalize_title(ref.title)
            
            # Check if the title matches
            if 'title' in item and item['title']:
                item_title = item['title'][0]
                normalized_item_title = normalize_title(item_title)
                
                # Check if the first author's family name matches
                author_match = False
                if 'author' in item and item['author'] and len(item['author']) > 0:
                    if 'family' in item['author'][0]:
                        author_match = ref.author == item['author'][0]['family']
                
                # Title matching with different levels of strictness
                match_type = classify_title_match(normalized_item_title, normalized_input_title)

                if match_type:
                    if author_match:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED,
                                                  explanation=f"DOI, author and title match Crossref record ({match_type} title match).")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.INVALID, 
                                                  explanation="DOI and title match Crossref record, but author does not match.")
                else:
                    return ReferenceCheckResult(status=ReferenceStatus.INVALID, 
                                              explanation="DOI matches Crossref record, but title does not match.")
            else:
                return ReferenceCheckResult(status=ReferenceStatus.INVALID, 
                                          explanation="DOI found in Crossref but no title available for comparison.")
        elif response.status_code == 404:
            # Fallback: resolve DOI via doi.org and try to parse metadata when Crossref doesn't have the record.
            doi_url = f"https://doi.org/{clean_doi}"
            try:
                doi_response = requests.get(
                    doi_url,
                    headers={"Accept": "application/vnd.citationstyles.csl+json"},
                    timeout=10,
                )
                if doi_response.status_code == 200:
                    item = doi_response.json()
                    normalized_input_title = normalize_title(ref.title)
                    item_title = item.get("title") or ""
                    normalized_item_title = normalize_title(item_title)

                    author_match = False
                    item_author = ""
                    authors = item.get("author") or []
                    if authors:
                        author = authors[0]
                        item_author = author.get("family") or author.get("literal") or ""
                        if item_author:
                            author_match = normalize_author_name(item_author) == normalize_author_name(ref.author)

                    match_type = classify_title_match(normalized_item_title, normalized_input_title)

                    if match_type:
                        if author_match:
                            return ReferenceCheckResult(
                                status=ReferenceStatus.VALIDATED,
                                explanation=f"DOI resolved via doi.org (CSL JSON); title and author match ({match_type})."
                            )
                        else:
                            return ReferenceCheckResult(
                                status=ReferenceStatus.INVALID,
                                explanation="DOI resolved via doi.org, but author does not match."
                            )
                    else:
                        return ReferenceCheckResult(
                            status=ReferenceStatus.INVALID,
                            explanation="DOI resolved via doi.org, but title does not match."
                        )
                else:
                    logging.warning(f"doi.org metadata request failed for '{clean_doi}' with status {doi_response.status_code}")

            except Exception as doi_error:
                logging.warning(f"Failed to resolve DOI '{clean_doi}' via doi.org: {doi_error}")

            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, 
                                      explanation="DOI not found in Crossref database or via doi.org metadata.")
        else:
            logging.warning(f"Crossref DOI API request failed with status code: {response.status_code}")
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, 
                                      explanation=f"Crossref DOI API request failed with status code: {response.status_code}")
    except Exception as e:
        logging.warning(f"Crossref DOI search failed for DOI '{ref.DOI}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, 
                                  explanation=f"Crossref DOI search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_crossref(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using the Crossref API, with retries and more robust matching. Returns ReferenceCheckResult."""
    # If DOI is provided, search by DOI first
    if ref.DOI and ref.DOI.strip():
        doi_result = search_doi_crossref(ref)
        if doi_result.status == ReferenceStatus.VALIDATED:
            return doi_result
        elif doi_result.status == ReferenceStatus.INVALID:
            # DOI found but doesn't match title/author - this is a clear invalid case
            # Only continue with title search if the failure was due to network issues or DOI not found
            if "DOI not found" not in doi_result.explanation and "failed" not in doi_result.explanation.lower():
                return doi_result
        # If DOI not found or there was a network error, continue with title search
    
    try:
        # Search by title
        params = {'query.title': ref.title, 'rows': 10}  # Increased rows to find more potential matches
        response = requests.get("https://api.crossref.org/works", params=params)

        if response.status_code == 200:
            items = response.json().get('message', {}).get('items', [])
            normalized_input_title = normalize_title(ref.title)
            
            # First pass: look for exact DOI matches
            ref_doi = ref.DOI.strip().lower() if ref.DOI else ''
            if ref_doi.startswith('https://doi.org/'):
                ref_doi = ref_doi[16:]
            elif ref_doi.startswith('http://doi.org/'):
                ref_doi = ref_doi[15:]
            elif ref_doi.startswith('doi:'):
                ref_doi = ref_doi[4:]
            
            for item in items:
                item_doi = item.get('DOI', '').strip().lower() if 'DOI' in item else ''
                
                # If we have both DOIs and they match exactly
                if ref_doi and item_doi and ref_doi == item_doi:
                    if 'author' in item and item['author'] and 'family' in item['author'][0]:
                        if ref.author == item['author'][0]['family']:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, 
                                                      explanation="Author, title and DOI match Crossref record.")
                        else:
                            return ReferenceCheckResult(status=ReferenceStatus.INVALID, 
                                                      explanation="DOI and title match Crossref record, but author does not match.")
                    else:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, 
                                                  explanation="Title and DOI match Crossref record.")
            
            # Second pass: look for title and author matches (when DOI was provided but didn't match)
            title_author_matches = []
            for item in items:
                if 'author' in item and item['author'] and 'family' in item['author'][0]:
                    if ref.author == item['author'][0]['family']:
                        # Check if the title matches
                        if 'title' in item and item['title']:
                            item_title = item['title'][0]
                            normalized_item_title = normalize_title(item_title)
                            
                            match_type = classify_title_match(normalized_item_title, normalized_input_title)

                            if match_type:
                                item_doi = item.get('DOI', '').strip().lower() if 'DOI' in item else ''
                                title_author_matches.append((item, match_type, item_doi))
            
            # If we found title and author matches
            if title_author_matches:
                # If DOI was provided in reference, check if any of the matches have the correct DOI
                if ref_doi:
                    for item, match_type, item_doi in title_author_matches:
                        if item_doi == ref_doi:
                            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, 
                                                      explanation=f"Author, title and DOI match Crossref record ({match_type} title match).")
                    
                    # If no exact DOI match found but we have title/author matches, this might be a case with multiple DOIs
                    # Return validated if we found a strong title/author match
                    best_match = title_author_matches[0]  # Take the first (presumably best) match
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, 
                                              explanation=f"Author and title match Crossref record ({best_match[1]} title match). Multiple DOI records may exist for this publication.")
                else:
                    # No DOI provided, return the best title/author match
                    best_match = title_author_matches[0]
                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, 
                                              explanation=f"Author and title match Crossref record ({best_match[1]} title match).")
            
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in Crossref.")
        else:
            logging.warning(f"Crossref API request failed with status code: {response.status_code}")
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Crossref API request failed with status code: {response.status_code}")
    except Exception as e:
        logging.warning(f"Crossref title search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Crossref title search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_arxiv(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title in arXiv, with error handling and retries."""
    try:
        # arXiv API endpoint
        url = "http://export.arxiv.org/api/query"
        
        # Search for the title - use double quotes around the title for exact match
        params = {
            'search_query': f'ti:"{ref.title}"',
            'max_results': 5
        }
        
        response = requests.get(url, params=params)

        if response.status_code == 200:
            # Parse the XML response - use 'lxml' parser for better compatibility
            soup = BeautifulSoup(response.content, 'lxml-xml')
            entries = soup.find_all('entry')
            
            if not entries:
                # Try a more flexible search if no exact matches
                params['search_query'] = f'all:{ref.title}'
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'lxml-xml')
                    entries = soup.find_all('entry')
            
            if not entries:
                return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
                
            normalized_input_title = normalize_title(ref.title)
            
            for entry in entries:
                title_tag = entry.find('title')
                if title_tag:
                    normalized_arxiv_title = normalize_title(title_tag.text.strip())

                    match_type = classify_title_match(normalized_arxiv_title, normalized_input_title)
                    if match_type:
                        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED,
                                                    explanation=f"Title match in arXiv ({match_type} match).")


                    # Check authors if titles are somewhat similar
                    if fuzz.ratio(normalized_arxiv_title, normalized_input_title) > 70:
                        author_tags = entry.find_all('author')
                        for author_tag in author_tags:
                            name_tag = author_tag.find('name')
                            if name_tag:
                                author_name = name_tag.text.strip()
                                # Extract last name
                                last_name = author_name.split()[-1]
                                if last_name.lower() == ref.author.lower():
                                    return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Author and similar title match in arXiv.")
                                
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No matching record found in arXiv.")
        else:
            logging.warning(f"arXiv API request failed with status code: {response.status_code}")
            return ReferenceCheckResult(
                status=ReferenceStatus.NOT_FOUND,
                explanation=f"arXiv API request failed with status code: {response.status_code}"
            )
        
    except Exception as e:
        logging.warning(f"arXiv search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"arXiv search failed: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_title_workshop_paper(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for workshop papers using Google Search directly."""
    try:
        # Check if it's likely a workshop paper from the reference text
        workshop_indicators = ['workshop', 'symposium', 'proc.', 'proceedings']
        is_likely_workshop = any(indicator in ref.bib.lower() for indicator in workshop_indicators)
        
        if not is_likely_workshop:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Not a workshop paper.")
            
        # Use Google search through the Google Gemini API with more specific prompt
        prompt = f"""
        Please search for this exact workshop paper and verify it exists:
        Title: {ref.title}
        Author: {ref.author}
        Year: {ref.year}
        
        This paper appears to be from a workshop or symposium. Check conferences, workshops, 
        and personal/university pages. Return 'True' only if you can find evidence this 
        specific workshop paper exists (exact title and author match). Return 'False' otherwise.
        Return only 'True' or 'False', without any additional explanation.
        """

        client = genai.Client(api_key=GOOGLE_API_KEY)
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt,
            config={
                'tools': [google_search_tool],
                'temperature': 0,
            },
        )

        if answers_true(response):
            return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Workshop paper found via Google search.")
        else:
            return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Workshop paper not found via Google search.")
            
    except Exception as e:
        logging.warning(f"Workshop paper search failed for title '{ref.title}': {e}")
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation=f"Workshop paper search failed: {e}")

MIN_TITLE_LEN_FOR_CONTAINMENT = 15  # Shorter normalized titles hit inside a page of text by accident.
MIN_PAGE_TEXT_FOR_MISMATCH = 60  # Below this the first page is effectively blank (scanned image).
PAGE_TEXT_FUZZ_THRESHOLD = 90  # Higher than TITLE_FUZZ_THRESHOLD: a title inside a page of text, not two titles.


def _read_capped(response) -> bytes:
    """Reads at most MAX_URL_DOWNLOAD_BYTES from a streamed response."""
    buf = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        buf += chunk[:MAX_URL_DOWNLOAD_BYTES - len(buf)]
        if len(buf) >= MAX_URL_DOWNLOAD_BYTES:
            break
    return bytes(buf)


def _extract_pdf_titles(data: bytes) -> Tuple[str, str]:
    """Returns (metadata title, first page text) for PDF bytes, empty strings when unreadable."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data), strict=False)
    except Exception as e:
        logging.info(f"Could not parse linked PDF: {e}")
        return "", ""

    # Two independent best-effort reads: a broken metadata dictionary must not cost us the page text.
    meta_title, first_page_text = "", ""
    try:
        if reader.metadata and reader.metadata.title:
            meta_title = str(reader.metadata.title).strip()
    except Exception as e:
        logging.info(f"Could not read linked PDF metadata: {e}")
    try:
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception as e:
        logging.info(f"Could not extract text from linked PDF: {e}")

    return meta_title, first_page_text


def match_pdf_title(ref: ReferenceExtraction, data: bytes) -> ReferenceCheckResult:
    """Compares the reference title against a downloaded PDF.

    Reading the document locally rather than handing it to a model costs no Gemini quota and keeps
    attacker-supplied document text out of a prompt that answers yes/no about the same reference.

    NOT_FOUND means the PDF carried no readable title evidence at all (scanned pages, an encrypted
    file, or a download truncated at MAX_URL_DOWNLOAD_BYTES), so the caller should fall back rather
    than report a mismatch it cannot support.
    """
    normalized_input = normalize_title(ref.title)
    if not normalized_input:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND,
                                    explanation="Reference has no title to compare against the linked PDF.")

    meta_title, page_text = _extract_pdf_titles(data)

    match_type = classify_title_match(normalize_title(meta_title), normalized_input)
    if match_type:
        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED,
                                    explanation=f"Linked PDF metadata title matches the reference title ({match_type} title match).")

    # Titles wrap across lines on a title page. normalize_title strips whitespace, so a wrapped
    # title becomes contiguous again and partial_ratio is the right test against the page text.
    normalized_page = normalize_title(page_text)
    if (len(normalized_input) >= MIN_TITLE_LEN_FOR_CONTAINMENT
            and fuzz.partial_ratio(normalized_input, normalized_page) > PAGE_TEXT_FUZZ_THRESHOLD):
        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED,
                                    explanation="Reference title found on the first page of the linked PDF.")

    # A metadata title is often authoring-tool junk ("Microsoft Word - draft.doc") or absent, so it
    # is good enough to confirm a match but never to reject one. Only a first page that actually
    # carries text justifies INVALID; a scanned or blank one reports no evidence and falls back.
    if len(normalized_page) >= MIN_PAGE_TEXT_FOR_MISMATCH:
        return ReferenceCheckResult(status=ReferenceStatus.INVALID,
                                    explanation="Linked PDF is reachable but its title does not match the reference title.")
    return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND,
                                explanation="Linked PDF carries no readable title (scanned, encrypted, or too large to parse).")


def with_google_fallback(ref: ReferenceExtraction, local_result: ReferenceCheckResult) -> ReferenceCheckResult:
    """Lets a successful Google search override a local mismatch before it is reported as invalid."""
    google_result = search_title_google(ref)
    if google_result.status == ReferenceStatus.VALIDATED:
        return google_result
    return local_result


def verify_pdf_url(ref: ReferenceExtraction, data: bytes) -> ReferenceCheckResult:
    """Verifies a reference against the PDF its URL points to, falling back to search."""
    pdf_result = match_pdf_title(ref, data)
    if pdf_result.status == ReferenceStatus.VALIDATED:
        return pdf_result
    if pdf_result.status == ReferenceStatus.NOT_FOUND:
        logging.info(f"No usable title in linked PDF: {ref.URL}. Falling back to Google search.")
        return search_title_google(ref)
    return with_google_fallback(ref, pdf_result)


def verify_url(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """
    Verifies if the title on the webpage at the given URL matches the reference title.
    """
    if not ref.URL:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No URL provided.")

    content, is_pdf, blocked = b"", False, False
    try:
        with requests.get(ref.URL, timeout=5, headers=DEFAULT_HTTP_HEADERS, stream=True) as response:
            blocked = response.status_code == 403
            if blocked:
                logging.info(f"Access denied (403) when fetching URL: {ref.URL}")
            else:
                response.raise_for_status()
                is_pdf = "application/pdf" in response.headers.get("Content-Type", "").lower()
                declared = response.headers.get("Content-Length", "")
                if is_pdf and declared.isdigit() and int(declared) > MAX_URL_DOWNLOAD_BYTES:
                    # A PDF truncated at the cap has no trailer left to parse, so downloading a
                    # known-oversized one buys nothing. Truncated HTML still yields its <title>.
                    logging.info(f"Linked PDF exceeds the {MAX_URL_DOWNLOAD_BYTES} byte cap, not downloaded: {ref.URL}")
                else:
                    content = _read_capped(response)
                    is_pdf = is_pdf or content.startswith(b"%PDF-")
        # The connection is released before any of the search fallbacks below make their own call.

        if blocked:
            google_result = search_title_google(ref)
            if google_result.status == ReferenceStatus.NOT_FOUND:
                return ReferenceCheckResult(
                    status=ReferenceStatus.NOT_FOUND,
                    explanation="Website blocked automated access (HTTP 403). Unable to confirm via direct fetch."
                )
            return google_result

        if is_pdf:
            return verify_pdf_url(ref, content)

        soup = BeautifulSoup(content, 'html.parser')
        title_tag = soup.find('title')

        if title_tag:
            webpage_title = title_tag.text.strip()
            normalized_webpage_title = normalize_title(webpage_title)
            normalized_input_title = normalize_title(ref.title)

            if normalized_webpage_title == normalized_input_title:
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (exact match).")
            elif normalized_input_title in normalized_webpage_title or normalized_webpage_title in normalized_input_title:  #robust matching
                return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Webpage title matches reference title (partial match).")
            logging.info(f"Webpage title '{webpage_title}' does not match reference '{ref.title}'. Falling back to Google search.")
            return with_google_fallback(ref, ReferenceCheckResult(
                status=ReferenceStatus.INVALID,
                explanation="URL reachable but webpage title does not match the reference title."
            ))
        else:
            logging.warning(f"No <title> tag found at URL: {ref.URL}")
            return search_title_google(ref)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        logging.warning(f"HTTP error accessing URL {ref.URL} (status {status_code}): {e}")
        return search_title_google(ref)
    except requests.exceptions.RequestException as e:
        logging.warning(f"Network error accessing URL {ref.URL}: {e}")
        return search_title_google(ref)  # Or consider raising the exception if you want to halt execution on URL errors.
    except Exception as e:
        logging.warning(f"Error processing URL {ref.URL}: {e}")
        return search_title_google(ref)


def search_title_google(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using Google Search and match using a LLM model."""

    prompt = f"""
    Please search for the reference on Google, compare with research results, and determine if it is genuine.\n
    Return 'True' only if a website with the the exact title and author is found. Otherwise, return 'False'.\n
    Return only 'True' or 'False', without any additional information.\n\n
    Author: {ref.author}\n
    Title: {ref.title}\n"""

    client = genai.Client(api_key=GOOGLE_API_KEY)
    google_search_tool = Tool(google_search=GoogleSearch())
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=prompt,
        config={
            'tools': [google_search_tool],
        },
    )

    if answers_true(response):
        return ReferenceCheckResult(status=ReferenceStatus.VALIDATED, explanation="Google search found matching reference.")
    else:
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="Google search did not find matching reference.")

def search_title(ref: ReferenceExtraction) -> ReferenceCheckResult:
    """Searches for a title using multiple methods."""
    if ref.type == "non_academic_website":
        return verify_url(ref)
    else:
        openalex_result = search_title_openalex(ref)
        if openalex_result.status != ReferenceStatus.NOT_FOUND:
            return openalex_result
        # First try Crossref
        crossref_result = search_title_crossref(ref)
        if crossref_result.status == ReferenceStatus.INVALID:
            return crossref_result
        if crossref_result.status == ReferenceStatus.VALIDATED:
            return crossref_result
        lobid_result = search_title_lobid(ref)
        if lobid_result.status != ReferenceStatus.NOT_FOUND:
            return lobid_result
        # For all academic papers, try arXiv as a fallback
        arxiv_result = search_title_arxiv(ref)
        if arxiv_result.status == ReferenceStatus.VALIDATED:
            return arxiv_result
        # Special check for workshop papers
        workshop_result = search_title_workshop_paper(ref)
        if workshop_result.status == ReferenceStatus.VALIDATED:
            return workshop_result
        # Fall back to Google Scholar
        scholar_result = search_title_scholarly(ref)
        if scholar_result.status == ReferenceStatus.VALIDATED:
            return scholar_result
        # If all fail, return the most informative NOT_FOUND
        for result in [crossref_result, lobid_result, arxiv_result, workshop_result, scholar_result]:
            if result.status == ReferenceStatus.NOT_FOUND:
                return result
        return ReferenceCheckResult(status=ReferenceStatus.NOT_FOUND, explanation="No evidence found in any source.")

# --- Main Workflow ---

def veriexcite(pdf_path: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Check references in a PDF. Returns:
    - count_verified: number of validated references
    - count_warning: number of warnings (invalid or not found)
    - list_warning: list of bib entries with warnings
    - list_explanations: list of explanations for each reference
    """
    # 1. Extract text from PDF and find bibliography
    full_text = extract_text_from_pdf(pdf_path)
    bib_text = extract_bibliography_section(full_text)
    # print("Extracted Bibliography Section:\n", bib_text, "\n")

    # 2. Split into individual references
    references = split_references(bib_text)
    # print(f"Found {len(references)} references.")

    # 3. Verify each reference
    count_verified, count_warning = 0, 0
    list_warning = []
    list_explanations = []

    for idx, ref in enumerate(references):
        result = search_title(ref)
        list_explanations.append(f"Reference: {ref.bib}\nStatus: {result.status.value}\nExplanation: {result.explanation}\n")
        if result.status == ReferenceStatus.VALIDATED:
            count_verified += 1
        else:
            count_warning += 1
            list_warning.append(ref.bib)
    return count_verified, count_warning, list_warning, list_explanations

def process_pdf_file(pdf_path: str) -> None:
    """Check a single PDF file."""
    count_verified, count_warning, list_warning, list_explanations = veriexcite(pdf_path)
    print(f"{count_verified} references verified, {count_warning} warnings.")
    if count_warning > 0:
        print("\nWarning List:\n")
        for item in list_warning:
            print(item)
    print("\nExplanation:\n")
    for explanation in list_explanations:
        print(explanation)
    return count_verified, count_warning, list_warning, list_explanations

def process_folder(folder_path: str) -> None:
    """Check all PDF files in a folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files in the folder.")

    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Checking file: {pdf_file}")
        count_verified, count_warning, list_warning, list_explanations = process_pdf_file(pdf_path)
        print("--------------------------------------------------")
        results.append({"File": pdf_file, "Found References": count_verified + count_warning, "Verified": count_verified,
                        "Warnings": count_warning, "Warning List": list_warning, "Explanation": list_explanations})
        pd.DataFrame(results).to_csv('VeriExCite results.csv', index=False, encoding='utf-8')
    print("Results saved to VeriExCite results.csv")


if __name__ == "__main__":
    ''' Set your Google Gemini API key here '''
    # Apply for a key at https://ai.google.dev/aistudio with hundreds requests per day for FREE
    GOOGLE_API_KEY = "YOUR_API_KEY"
    set_google_api_key(GOOGLE_API_KEY)

    ''' Example usage #1: check a single PDF file '''
    # pdf_path = "path/to/your/paper.pdf"
    # process_pdf_file(pdf_path)

    ''' Example usage #2: check all PDF files in a folder '''
    # Please replace the folder path to your directory containing the PDF files.
    folder_path = "path/to/your/folder"
    process_folder(folder_path)
