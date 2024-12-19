# api/scraping.py
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

# api/urls.py

SCRAPE_URLS = {
    "driver_license_application": "https://www.dmv.virginia.gov/licenses-ids/license/applying",
    "state_id_application": "https://dds.georgia.gov/georgia-licenseid/new-licenseid/apply-new-ga-license",
    "vehicle_registration": "https://www.usa.gov/state-motor-vehicle-services"
}


def scrape_html(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def scrape_pdf(url: str) -> str:
    response = requests.get(url)
    pdf = PdfReader(response.content)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def scrape_documents():
    for doc_name, url in SCRAPE_URLS.items():
        if url.endswith(".pdf"):
            text = scrape_pdf(url)
        else:
            text = scrape_html(url)
        # Process text, clean it, and store it in FAISS
        print(f"Scraped text from {doc_name}: {text[:200]}...")  # Previewing the text
