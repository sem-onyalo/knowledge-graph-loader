import csv
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pyopenie import OpenIE5
from queue import Empty, Queue
from spacy.lang.en import English
from time import sleep
from typing import List

ENCODING = "utf-8"
DATA_DIRECTORY = "./data"
CACHE_DIRECTORY = "cache/"
CACHE_CONNECTIONS_FILE = ".entity_connections"
CONNECTION_BUILDER_THREADS = 5
SENTENCE_QUEUE_WAIT_TIMEOUT = 5
RELATIONSHIP_EXTRACTION_SERVICE_RETRIES = 5
RELATIONSHIP_EXTRACTION_SERVICE_TIMEOUT = 3
RELATIONSHIP_EXTRACTION_SERVICE_URL = 'http://localhost:8000'

class Document:
    file_name:str
    sentences:list
    def __init__(self, file_name, sentences) -> None:
        self.file_name = file_name
        self.sentences = sentences

class DocumentSentence:
    document:Document
    sentence:str
    def __init__(self, document, sentence) -> None:
        self.document = document
        self.sentence = sentence

class EntityConnection:
    from_entity:str
    to_entity:str
    relationship:str
    confidence:float
    file_name:str
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            other:EntityConnection = __o
            return (self.from_entity == other.from_entity
                and self.to_entity == other.to_entity
                and self.relationship == other.relationship
                and self.confidence == other.confidence
                and self.file_name == other.file_name)
        else:
            return False

nlp:English = None
extractor:OpenIE5 = None
sentence_queue:Queue = None
connection_list:List[EntityConnection] = None

def init_logger(level=logging.DEBUG):
    logging.basicConfig(
        format="[%(asctime)s]\t[%(levelname)s]\t[%(name)s]\t%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

def init_cache():
    cache_dir = os.path.join(DATA_DIRECTORY, CACHE_DIRECTORY)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

def init_sentencizer() -> None:
    global nlp
    nlp = English()
    nlp.add_pipe("sentencizer")

def init_sentence_queue() -> None:
    global sentence_queue
    sentence_queue = Queue()

def init_connection_list() -> None:
    global connection_list
    cache_connections = get_cache_connections()
    connection_list = cache_connections if cache_connections != None else list()

def init_relationship_extractor() -> None:
    global extractor
    extractor = OpenIE5(RELATIONSHIP_EXTRACTION_SERVICE_URL)

def cache_connections() -> None:
    path = os.path.join(DATA_DIRECTORY, CACHE_DIRECTORY, CACHE_CONNECTIONS_FILE)
    with open(path, mode="w", encoding=ENCODING) as fd:
        writer = csv.writer(fd)
        for c in connection_list:
            row = [c.from_entity, c.to_entity, c.relationship, c.confidence, c.file_name]
            # TODO: fix extra CRLF at end of line
            writer.writerow(row)

def get_cache_connections() -> List[EntityConnection]:
    FROM_ENTITY_IDX = 0
    TO_ENTITY_IDX = 1
    RELATIONSHIP_IDX = 2
    CONFIDENCE_IDX = 3
    FILE_NAME_IDX = 4
    connections = list()
    path = os.path.join(DATA_DIRECTORY, CACHE_DIRECTORY, CACHE_CONNECTIONS_FILE)
    if os.path.isfile(path):
        with open(path, mode="r", encoding=ENCODING) as fd:
            reader = csv.reader(fd)
            for row in reader:
                if len(row) == 0:
                    continue

                connection = EntityConnection()
                connection.from_entity = row[FROM_ENTITY_IDX]
                connection.to_entity = row[TO_ENTITY_IDX]
                connection.relationship = row[RELATIONSHIP_IDX]
                connection.confidence = float(row[CONFIDENCE_IDX])
                connection.file_name = row[FILE_NAME_IDX]
                connections.append(connection)
        return connections

def extract_sentences_from_data(data) -> list:
    document = nlp(data)
    return [s.text for s in document.sents]

def extract_data_from_file(file_path) -> str:
    with open(file_path, encoding=ENCODING) as fd:
        data = fd.read()
    return data

def build_documents_from_files(data_files) -> List[Document]:
    documents = list()
    for data_file in data_files:
        data = extract_data_from_file(data_file)
        sentences = extract_sentences_from_data(data)
        documents.append(Document(data_file, sentences))
    return documents

def build_connection_from_extraction(extraction:dict, document:Document) -> None:
    if len(extraction["extraction"]["arg2s"]) > 0:
        connection = EntityConnection()
        connection.from_entity = extraction["extraction"]["arg1"]["text"]
        # TODO: add logic for handling multiple arg2s
        connection.to_entity = extraction["extraction"]["arg2s"][0]["text"]
        connection.relationship = extraction["extraction"]["rel"]["text"]
        connection.confidence = float(extraction["confidence"])
        connection.file_name = os.path.basename(document.file_name.replace("\\", os.sep))
        connection_list.append(connection)

def build_connections_from_document(threadId:str) -> None:
    logging.info(f"{threadId} thread started")
    sentences_processed = 0
    while True:
        try:
            docSentence:DocumentSentence = sentence_queue.get(timeout=SENTENCE_QUEUE_WAIT_TIMEOUT)
        except Empty:
            logging.info(f"{threadId} thread exiting, sentence queue empty")
            return sentences_processed, threadId

        got_extractions = False
        current_try = RELATIONSHIP_EXTRACTION_SERVICE_RETRIES
        while current_try > 0:
            try:
                extractions = extractor.extract(docSentence.sentence)
                got_extractions = True
                sentences_processed += 1
                break
            except Exception as e:
                logging.debug(f"{threadId} thread service exception on try {current_try}: {e}")
                sleep(RELATIONSHIP_EXTRACTION_SERVICE_TIMEOUT)
                current_try -= 1

        if not got_extractions:
            logging.error(f"{threadId} thread skipping item, could not process sentence: {docSentence.sentence}")
            continue

        for extraction in extractions:
            build_connection_from_extraction(extraction, docSentence.document)

def build_connections_from_documents(documents:List[Document]) -> List[EntityConnection]:
    if len(connection_list) > 0:
        logging.info("Skipping build connections, list populated by cache")
        return

    sentences_count = 0
    for document in documents:
        for sentence in document.sentences:
            sentence_queue.put(DocumentSentence(document, sentence))
            sentences_count += 1

    sentences_processed = 0
    with ThreadPoolExecutor(max_workers=CONNECTION_BUILDER_THREADS) as executor:
        threadIds = [uuid.uuid4() for _ in range(CONNECTION_BUILDER_THREADS)]
        futures = executor.map(build_connections_from_document, threadIds)
        for future in futures:
            logging.debug(f"Thread result {future}")
            sentences_processed += int(future[0])

    logging.info(f"{sentences_processed} of {sentences_count} sentences processed")

    cache_connections()

def filter_connections():
    length_before = len(connection_list)
    for connection in connection_list:
        if connection.from_entity in nlp.Defaults.stop_words:
            connection_list.remove(connection)
    length_after = len(connection_list)
    logging.info(f"{length_before - length_after} entity connections removed from list")

def main():
    init_logger()

    init_cache()

    init_sentencizer()

    init_sentence_queue()

    init_connection_list()

    init_relationship_extractor()

    data_files = [os.path.join(DATA_DIRECTORY, f) for f in os.listdir(DATA_DIRECTORY) if os.path.isfile(os.path.join(DATA_DIRECTORY, f))]

    documents = build_documents_from_files(data_files)

    build_connections_from_documents(documents)

    filter_connections()

if __name__ == "__main__":
    main()