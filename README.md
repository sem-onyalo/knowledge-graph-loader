# Knowledge Graph Loader Using NLP Relationship Extraction

This repository using NLP relationship extraction to extract entity-relationship combinations from a body of text and loads it into a graph database.

## Setup

### Graph Database

This repository loads data into a [neo4j](https://neo4j.com/) database. To use this code you'll need to download, install and setup [Neo4j Desktop](https://neo4j.com/download/).

### NLP Relationship Extraction

This repository uses the [Open IE](https://www.github.com/dair-iitd/OpenIE-standalone/) project to extract entities and relationships from text. To use this code you'll need to clone and follow the setup instructions on the Open IE GitHub repository page and run the project [as an HTTP service](https://github.com/dair-iitd/OpenIE-standalone/#running-as-http-server).

### Source Data

This repository uses the text files in the [data](./data) directory as its data source, which is a manual extract from ten [White House Press Briefing](https://www.whitehouse.gov/briefing-room/press-briefings/) documents. If you wish to use another data source simply replace the text files in the [data](./data) directory.

#### Data Caching

Results from Open IE are [cached to a file](./data/cache/entity_connections.cache). If you change the data in the [data](./data) directory after this file is created you'll need to delete this file.

### Repository Setup

Run the following commands to install the dependencies and run the code.

```
python -m venv env

source venv/bin/activate # or for Windows: .\env\Scripts\activate

pip install -r requirements.txt
```

## Running

To run this code use the following command.

```
python main.py
```

Once the data is loaded into neo4j you can verify the load with the following [cypher query](https://neo4j.com/docs/cypher-manual/current/).

```
MATCH (n:Entity) RETURN n LIMIT 1000
```

To explore the data using [neo4j's graph algorithms](https://neo4j.com/docs/graph-data-science/current/algorithms/) you'll first need to create a graph catalog like shown below (see [the page on graph management](https://neo4j.com/docs/graph-data-science/current/management-ops/) for more details).

```
CALL gds.graph.create('knowledge-graph-catalog', 'Entity', 'RELATION', { relationshipProperties:'confidence'})
```

Once your graph catalog is created you can run the graph algorithm of your choice. For example, 

```
CALL gds.pageRank.stream('knowledge-graph-catalog')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name ASC
```
