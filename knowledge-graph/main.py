import logging
import os
from pyopenie import OpenIE5
from spacy.lang.en import English
from typing import List

DATA_FILES_PATH = "./data"
RELATIONSHIP_EXTRACTION_SERVICE_URL = 'http://localhost:8000'

nlp = None
extractor = None

class Document:
    file_name:str
    sentences:list
    def __init__(self, file_name, sentences) -> None:
        self.file_name = file_name
        self.sentences = sentences

class EntityConnection:
    from_entity:str
    to_entity:str
    relationship:str
    confidence:float
    file_name:str

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )

def init_sentencizer() -> None:
    global nlp
    nlp = English()
    nlp.add_pipe("sentencizer")

def init_relationship_extractor() -> None:
    global extractor
    extractor = OpenIE5(RELATIONSHIP_EXTRACTION_SERVICE_URL)

def extract_sentences_from_data(data) -> list:
    document = nlp(data)
    return [s.text for s in document.sents]

def extract_data_from_file(file_path) -> str:
    with open(file_path, encoding="utf-8") as fd:
        data = fd.read()
    return data

def build_documents_from_files(data_files) -> List[Document]:
    documents = list()
    for data_file in data_files:
        data = extract_data_from_file(data_file)
        sentences = extract_sentences_from_data(data)
        documents.append(Document(data_file, sentences))
    return documents

def build_connections_from_extraction(extraction, document:Document, connections:List[EntityConnection]) -> EntityConnection:
    if len(extraction["extraction"]["arg2s"]) > 0:
        connection = EntityConnection()
        connection.from_entity = extraction["extraction"]["arg1"]["text"]
        # TODO: add logic for handling multiple arg2s
        connection.to_entity = extraction["extraction"]["arg2s"][0]["text"]
        connection.relationship = extraction["extraction"]["rel"]["text"]
        connection.confidence = float(extraction["confidence"])
        connection.file_name = document.file_name
        connections.append(connection)
    return connections

def build_connections_from_documents(documents:List[Document]) -> List[EntityConnection]:
    connections = list()
    sentences_count = 0
    sentences_processed_count = 0
    for document in documents:
        for sentence in document.sentences:
            extractions = extractor.extract(sentence)
            for extraction in extractions:
                connections = build_connections_from_extraction(extraction, document, connections)
                sentences_processed_count += 1
            sentences_count += 1
    logging.debug(f"Processed {sentences_processed_count} of {sentences_count} sentences")
    return connections

def main():
    init_sentencizer()

    init_relationship_extractor()

    data_files = [os.path.join(DATA_FILES_PATH, f) for f in os.listdir(DATA_FILES_PATH)]

    documents = build_documents_from_files(data_files)

    connections = build_connections_from_documents(documents)

if __name__ == "__main__":
    main()