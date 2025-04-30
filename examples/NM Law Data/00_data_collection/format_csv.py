import os
import json
import re
import pandas as pd

class JSONToCSVProcessor:
    """
    A class to process JSON files into a single CSV, handling multiple variations
    of how JSON data may be structured.

    Usage:
        processor = JSONToCSVProcessor(root_path, output_path)
        # Choose a processor_type (1, 2, or 3) corresponding to the different
        # ways you'd like to handle the JSON data.
        df = processor.process_json_files_to_csv(processor_type=1)
    """

    def __init__(self, root_path, output_path):
        """
        :param root_path: The directory path where JSON files are located.
        :param output_path: The path where the resulting CSV will be saved.
        """
        self.root_path = root_path
        self.output_path = output_path

    def process_json_files_to_csv(self, processor_type=1):
        """
        Processes JSON files into a CSV file.

        processor_type=1 (First/Third version in your code):
            1) 'chapter' -- 'article' -- 'section' 
               - Excludes "repealed" items
               - Uses regex to extract citation (e.g., '§ X (YYYY)')
               - Excludes multiple sections (e.g., ' to ' in the citation)
               - Abstract is a concatenation of 'paragraphs'

        processor_type=2 (Second version in your code):
            2) 'article' -- 'section'
               - Excludes "repealed" items
               - Citation is processed by splitting 'NM Const art...' 
               - Abstract is the first string in the 'text' list

        :param processor_type: int
            Which JSON-processing variant to use. Defaults to 1.
        :return: pd.DataFrame
        """

        records = []

        # Traverse the directory
        for subdir, _, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(subdir, file)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # ---- Processor Type 1 or 3 logic ----
                        if processor_type in (1, 3):
                            # Build title from 'chapter', 'article', 'section'
                            chapter = data.get('chapter', '')
                            article = data.get('article', '')
                            section = data.get('section', '')
                            title_parts = [chapter, article, section]
                            title = ' -- '.join(filter(None, title_parts))

                            # Exclude repealed laws
                            if 'repealed' in title.lower():
                                continue

                            # Attempt to extract citation via regex
                            citation_raw = data.get('citation', '')
                            citation_match = re.search(r'§ (.*?) \(\d{4}\)', citation_raw)
                            citation = citation_match.group(1) if citation_match else ''

                            # If there's ' to ' in the citation, assume it's multiple sections (exclude)
                            if ' to ' in citation.lower():
                                continue

                            # Combine paragraphs into a single abstract
                            paragraphs = data.get('paragraphs', [])
                            abstract = ' '.join(paragraphs)

                            # Retrieve the URL
                            url = data.get('url', '')

                            # Add record to list
                            records.append({
                                'title': title,
                                'abstract': abstract,
                                'citation': citation,
                                'url': url
                            })

                        # ---- Processor Type 2 logic ----
                        elif processor_type == 2:
                            # Build title from 'article', 'section'
                            article = data.get('article', '')
                            section = data.get('section', '')
                            title_parts = [article, section]
                            title = ' -- '.join(filter(None, title_parts))

                            # Exclude repealed laws
                            if 'repealed' in title.lower():
                                continue

                            # Clean citation by splitting out 'NM Const art'
                            citation_raw = data.get('citation', '')
                            if 'NM Const art' in citation_raw:
                                # Example transform: 'NM Const artX § Y' -> 'X-Y'
                                citation = citation_raw.split('NM Const art')[-1].replace(' § ', '-')
                            else:
                                citation = citation_raw

                            # Abstract from the first element in 'text'
                            text_list = data.get('text', [])
                            if text_list and len(text_list) > 0:
                                abstract = text_list[0]
                            else:
                                abstract = ''

                            url = data.get('url', '')

                            # Add record to list
                            records.append({
                                'title': title,
                                'abstract': abstract,
                                'citation': citation,
                                'url': url
                            })

        # Convert records to DataFrame and save to CSV
        df = pd.DataFrame(records)
        df.to_csv(self.output_path, index=False)
        print(f"Data has been written to {self.output_path}")
        return df