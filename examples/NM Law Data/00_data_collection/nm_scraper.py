import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from requests.exceptions import HTTPError
import json
import datetime

class NewMexicoScraper:
    def __init__(self, base_url, output_dir, headers, saved_urls_file):
        """
        :param base_url: Overall base URL for New Mexico legal materials.
                         For cases and opinions use: "https://law.justia.com/new-mexico/"
        :param output_dir: Base directory to save all downloads.
        :param headers: HTTP headers for requests.
        :param saved_urls_file: File to store already processed URLs.
        """
        self.base_url = base_url  # e.g., "https://law.justia.com/new-mexico/"
        self.output_dir = output_dir
        self.headers = headers
        self.saved_urls_file = saved_urls_file
        os.makedirs(self.output_dir, exist_ok=True)
    
    # -------------------------
    # Helper Methods
    # -------------------------
    def get_type_output_dir(self, type_name):
        """Return a base directory for a given document type."""
        dir_path = os.path.join(self.output_dir, type_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_soup(self, url, retries=3):
        """Fetch content from a URL using retry logic and return a BeautifulSoup object."""
        for i in range(retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except HTTPError as http_err:
                print(f"HTTP error occurred: {http_err} for url: {url}")
                if response.status_code == 403:
                    print("Access forbidden. Retrying with backoff...")
                time.sleep(2 ** i)
            except Exception as err:
                print(f"Other error occurred: {err} for url: {url}")
                break
        return None

    def save_json(self, path, data):
        """Save data to a JSON file in the appropriate directory."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def load_saved_urls(self):
        """Load saved URLs from file to avoid reprocessing."""
        if os.path.exists(self.saved_urls_file):
            with open(self.saved_urls_file, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        return set()

    def save_url(self, url):
        """Append a processed URL to the saved URLs file."""
        with open(self.saved_urls_file, 'a', encoding='utf-8') as file:
            file.write(url + '\n')

    def archive_saved_urls(self):
        """Archive the saved URLs file with a timestamp."""
        if os.path.exists(self.saved_urls_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(self.saved_urls_file, os.path.join(self.output_dir, f"saved_urls_archive_{timestamp}.txt"))
    
    # -------------------------
    # Methods for Statutes (Laws) Scraping by Year
    # -------------------------
    def scrape_section(self, section_url, titles, overwrite=False):
        """Scrape text from a law section and save as JSON under the 'statutes' directory,
           inside a subdirectory named after the chapter number."""
        print(f"Processing section: {section_url}")
        saved_urls = self.load_saved_urls()
        if section_url in saved_urls and not overwrite:
            print(f"Section {section_url} already saved. Skipping...")
            return
        
        soup = self.get_soup(section_url)
        if soup:
            citation_div = soup.find('div', class_='citation')
            if citation_div:
                citation = citation_div.find('span').get_text(strip=True)
            else:
                citation = "Unknown_Citation"
            
            codes_content = soup.find(id="codes-content")
            if codes_content:
                paragraphs = [p.get_text(strip=True) for p in codes_content.find_all('p') if p.get_text(strip=True)]
                citation_filename = citation.replace(' ', '_').replace('/', '-')
                # Create a directory for the statutes under the chapter number.

                match = re.search(r'chapter-([^/]+)/', section_url)
                dir_name = match.group(1) if match else "unknown"
                chapter_dir = os.path.join(self.get_type_output_dir("statutes"), dir_name)

                os.makedirs(chapter_dir, exist_ok=True)
                file_path = os.path.join(chapter_dir, citation_filename + '.json')
                data = {
                    'chapter': titles.get('chapter', ''),
                    'article': titles.get('article', ''),
                    'section': titles.get('section', ''),
                    "citation": citation,
                    "url": section_url,
                    "paragraphs": paragraphs
                }
                self.save_json(file_path, data)
                self.save_url(section_url)
            else:
                print(f"No content found in section {citation}")
        else:
            print(f"Failed to fetch section {section_url}")

    def scrape_article_laws(self, article_url, titles, overwrite=False):
        """Scrape all sections within an article for the laws."""
        soup = self.get_soup(article_url)
        if soup:
            for section in soup.select('a'):
                section_href = section.get('href')
                if section_href and 'section' in section_href:
                    section_url = urljoin(self.base_url, section_href)
                    section_text = section.get_text(strip=True).split(' - ')[-1]
                    titles['section'] = section_text
                    self.scrape_section(section_url, titles, overwrite)
        else:
            print(f"Failed to fetch article from {article_url}")

    def scrape_chapter(self, chapter_url, titles, overwrite=False):
        """Scrape all articles within a chapter for the laws."""
        soup = self.get_soup(chapter_url)
        if soup:
            for article in soup.select('a'):
                article_href = article.get('href')
                if article_href and 'article' in article_href:
                    article_url = urljoin(self.base_url, article_href)
                    article_text = article.get_text(strip=True).split(' - ')[-1]
                    titles['article'] = article_text
                    self.scrape_article_laws(article_url, titles, overwrite)
        else:
            print(f"Failed to fetch chapter {titles.get('chapter', '')}")

    def scrape_laws_by_year(self, year, overwrite=False):
        """
        Scrape all statutes (codes) for a given year.
        URL structure: "https://law.justia.com/codes/new-mexico/<year>/"
        """
        codes_base = "https://law.justia.com/codes/new-mexico/"
        laws_url = urljoin(codes_base, f"{year}/")
        soup = self.get_soup(laws_url)
        if not soup:
            print(f"Listing URL for statutes for year {year} not found. Skipping...")
            return
        for chapter in soup.select('a'):
            chapter_href = chapter.get('href')
            if chapter_href and 'chapter' in chapter_href:
                chapter_url = urljoin(laws_url, chapter_href)
                # Here we assume the chapter text (number) is available;
                # adjust splitting or regex as needed.
                chapter_text = chapter.get_text(strip=True).split(' - ')[-1]
                titles = {'chapter': chapter_text}
                self.scrape_chapter(chapter_url, titles, overwrite)
    
    # -------------------------
    # Methods for Constitution Scraping
    # -------------------------
    def get_citation_constitution(self, soup):
        """Extract the citation in the 'Universal Citation' format for constitution pages."""
        citation_div = soup.find('div', class_='has-margin-bottom-20')
        if citation_div and 'Universal Citation' in citation_div.text:
            return citation_div.find('a').get_text(strip=True)
        return "Unknown Citation"
    
    def scrape_content(self, url, filename, article=None, section=None, overwrite=False, citation=None):
        """Generic function to scrape and save content from a URL under the 'const' directory."""
        saved_urls = self.load_saved_urls()
        if url in saved_urls and not overwrite:
            print(f"{filename.capitalize()} already saved. Skipping...")
            return
        soup = self.get_soup(url)
        if soup:
            if not citation:
                citation = self.get_citation_constitution(soup)
            codes_content = soup.find(id="codes-content")
            if codes_content:
                paragraphs = [codes_content.get_text(strip=True)]
                cite_file_name = citation.replace(" ", "_").replace('/', '-')
                base_dir = self.get_type_output_dir("const")
                file_path = os.path.join(base_dir, f"{cite_file_name}.json")
                data = {
                    'article': article,
                    'section': section,
                    'citation': citation,
                    "url": url,
                    "text": paragraphs
                }
                self.save_json(file_path, data)
                self.save_url(url)
        else:
            print(f"Failed to fetch {filename} from {url}")
    
    def scrape_preamble(self, overwrite=False, citation='preamble'):
        """Scrape the preamble of the constitution under the 'const' directory."""
        preamble_url = "https://law.justia.com/constitution/new-mexico/preamble/"
        self.scrape_content(preamble_url, 'preamble', overwrite=overwrite, citation=citation)
    
    def scrape_article_constitution(self, article_url, titles, overwrite=False):
        """Scrape sections within an article of the constitution."""
        soup = self.get_soup(article_url)
        if soup:
            codes_content = soup.find(id="codes-content")
            if codes_content:
                self.scrape_content(article_url, titles['article'], article=titles['article'], overwrite=overwrite)
            else:
                for section in soup.find_all('a', href=True):
                    if 'section' in section['href']:
                        section_url = urljoin(self.base_url, section['href'])
                        section_title = section.get_text(strip=True)
                        titles['section'] = section_title
                        self.scrape_content(section_url, section_title.replace(' ', '_'),
                                            article=titles['article'], section=section_title, overwrite=overwrite)
        else:
            print(f"Failed to fetch article {titles.get('article', '')}")
    
    def scrape_constitution(self, overwrite=False):
        """Main function to scrape the preamble and all articles of the constitution."""
        constitution_url = "https://law.justia.com/constitution/new-mexico/"
        if overwrite:
            self.archive_saved_urls()
        self.scrape_preamble(overwrite, citation='preamble')
        soup = self.get_soup(constitution_url)
        if soup:
            for article in soup.find_all('a', href=True):
                if 'article' in article['href']:
                    article_url = urljoin(constitution_url, article['href'])
                    article_title = article.get_text(strip=True)
                    titles = {'article': article_title}
                    self.scrape_article_constitution(article_url, titles, overwrite)
        else:
            print("Failed to fetch the main page for the constitution.")
    
    # -------------------------
    # Methods for Court Decisions (Supreme & Appeals)
    # -------------------------
    def scrape_listing(self, listing_url, decision_type, overwrite=False):
        """
        Generic function to scrape a listing page with pagination.
        :param listing_url: URL of the listing page for a given year.
        :param decision_type: 'supreme-court' or 'court-of-appeals'
        """
        while listing_url:
            soup = self.get_soup(listing_url)
            if not soup:
                print(f"Failed to fetch listing page: {listing_url}")
                break
            for link in soup.select('a'):
                href = link.get('href')
                if href and decision_type in href and href.endswith('.html'):
                    decision_url = urljoin(self.base_url, href)
                    if decision_type == 'supreme-court':
                        self.scrape_supreme_court_decision(decision_url, overwrite)
                    elif decision_type == 'court-of-appeals':
                        self.scrape_court_of_appeals_decision(decision_url, overwrite)
            next_page = soup.find('a', text='Next')
            if next_page:
                listing_url = urljoin(self.base_url, next_page.get('href'))
            else:
                listing_url = None

    def scrape_supreme_court_decision(self, decision_url, overwrite=False):
        """Scrape a single Supreme Court decision and save it under 'supreme/<year>'."""
        print(f"Processing Supreme Court decision: {decision_url}")
        saved_urls = self.load_saved_urls()
        if decision_url in saved_urls and not overwrite:
            print(f"Decision {decision_url} already saved. Skipping...")
            return
        soup = self.get_soup(decision_url)
        if not soup:
            print(f"Failed to fetch decision page: {decision_url}")
            return
        case_name_tag = soup.find('h1')
        case_name = case_name_tag.get_text(strip=True) if case_name_tag else "Unknown_Case"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        parsed_url = urlparse(decision_url)
        path_parts = parsed_url.path.split('/')
        year = "UnknownYear"
        try:
            idx = path_parts.index('supreme-court')
            year_candidate = path_parts[idx + 1]
            if year_candidate.isdigit():
                year = year_candidate
        except (ValueError, IndexError):
            pass
        if not case_name.endswith(year):
            case_name = f"{case_name}_{year}"
        base_dir = os.path.join(self.get_type_output_dir("supreme"), year)
        os.makedirs(base_dir, exist_ok=True)
        filename = case_name.replace(' ', '_').replace('/', '-') + '.json'
        file_path = os.path.join(base_dir, filename)
        data = {
            "case_name": case_name,
            "url": decision_url,
            "paragraphs": paragraphs
        }
        self.save_json(file_path, data)
        self.save_url(decision_url)
    
    def scrape_court_of_appeals_decision(self, decision_url, overwrite=False):
        """Scrape a single Court of Appeals decision and save it under 'appeals/<year>'."""
        print(f"Processing Court of Appeals decision: {decision_url}")
        saved_urls = self.load_saved_urls()
        if decision_url in saved_urls and not overwrite:
            print(f"Decision {decision_url} already saved. Skipping...")
            return
        soup = self.get_soup(decision_url)
        if not soup:
            print(f"Failed to fetch decision page: {decision_url}")
            return
        case_name_tag = soup.find('h1')
        case_name = case_name_tag.get_text(strip=True) if case_name_tag else "Unknown_Case"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        parsed_url = urlparse(decision_url)
        path_parts = parsed_url.path.split('/')
        year = "UnknownYear"
        try:
            idx = path_parts.index('court-of-appeals')
            year_candidate = path_parts[idx + 1]
            if year_candidate.isdigit():
                year = year_candidate
        except (ValueError, IndexError):
            pass
        if not case_name.endswith(year):
            case_name = f"{case_name}_{year}"
        base_dir = os.path.join(self.get_type_output_dir("appeals"), year)
        os.makedirs(base_dir, exist_ok=True)
        filename = case_name.replace(' ', '_').replace('/', '-') + '.json'
        file_path = os.path.join(base_dir, filename)
        data = {
            "case_name": case_name,
            "url": decision_url,
            "paragraphs": paragraphs
        }
        self.save_json(file_path, data)
        self.save_url(decision_url)

    def scrape_supreme_court_by_year(self, year, overwrite=False):
        """
        Scrape all Supreme Court decisions for a given year.
        URL structure: "https://law.justia.com/new-mexico/cases/supreme-court/<year>/"
        """
        listing_url = urljoin(self.base_url, f"cases/supreme-court/{year}/")
        soup = self.get_soup(listing_url)
        if not soup:
            print(f"Listing URL for Supreme Court decisions for year {year} not found. Skipping...")
            return
        self.scrape_listing(listing_url, "supreme-court", overwrite)

    def scrape_court_of_appeals_by_year(self, year, overwrite=False):
        """
        Scrape all Court of Appeals decisions for a given year.
        URL structure: "https://law.justia.com/new-mexico/cases/court-of-appeals/{year}/"
        """
        listing_url = urljoin(self.base_url, f"cases/court-of-appeals/{year}/")
        soup = self.get_soup(listing_url)
        if not soup:
            print(f"Listing URL for Court of Appeals decisions for year {year} not found. Skipping...")
            return
        self.scrape_listing(listing_url, "court-of-appeals", overwrite)

    def scrape_all_supreme_court(self, years, overwrite=False):
        """Scrape Supreme Court decisions for all specified years."""
        for year in years:
            self.scrape_supreme_court_by_year(year, overwrite)

    def scrape_all_court_of_appeals(self, years, overwrite=False):
        """Scrape Court of Appeals decisions for all specified years."""
        for year in years:
            self.scrape_court_of_appeals_by_year(year, overwrite)