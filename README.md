Here's a complete README for the setup, including details for the `app.py`, Docker setup, and the `widoco.sh` script.

---

# Ontology Question Answering System - README

This system is designed to load an ontology into a Fuseki server, query the ontology using SPARQL, and convert natural language questions into SPARQL queries with the help of an LLM. The application also includes documentation generation using Widoco.

## Components

### 1. **app.py**
The main Streamlit application that handles the ontology upload, querying, and interaction with the LLM.

#### Key Features:
- Upload an ontology file (`.owl`, `.rdf`, `.ttl`).
- Extract and load ontology into a Fuseki server.
- Generate embeddings for ontology elements (e.g., classes, object properties).
- Convert user queries in natural language to SPARQL queries using an LLM.
- Execute the SPARQL queries against the Fuseki server.
- Display the query results.

#### Required Libraries:
- `streamlit`
- `rdflib`
- `requests`
- `pandas`
- `sentence-transformers`
- `torch`
- `faiss` (if vector store features are needed)
  
#### Running the Application
The app will start on port 8501 by default. You can access it by navigating to `http://localhost:8501` on your browser.

### 2. **docker-compose.yml**
The Docker Compose file sets up multiple services, including the Fuseki server, the Streamlit app, NGINX for serving static files, and MongoDB for WebProtege.

#### Services:
- **fuseki**: Runs Apache Jena Fuseki for hosting the RDF database.
- **streamlit**: Runs the Streamlit app (`app.py`).
- **nginx**: Serves static files for WebProtege.
- **wpmongo**: MongoDB for WebProtege.
- **webprotege**: WebProtege service for ontology management.

#### Usage:
1. Build and run the services using:
   ```bash
   docker-compose up --build
   ```

2. After running, the following services will be available:
   - Streamlit app on `http://localhost:8501`.
   - WebProtege on `http://localhost:5000`.
   - Fuseki server on `http://localhost:3030`.

### 3. **widoco.sh**
A shell script for generating ontology documentation using Widoco. The documentation will be generated from an ontology file (`wirr_project3.rdf`) located in the `data/ontologies` folder and saved to `data/target/generated-doc`.

#### Requirements:
- Docker installed on your system.

#### Usage:
1. Ensure that `wirr_project3.rdf` is located in the `data/ontologies` directory.
2. Run the script using:
   ```bash
   ./widoco.sh
   ```
3. The documentation will be generated in the `data/target/generated-doc` folder.

---

### Troubleshooting

1. **Streamlit App Issues:**
   - Ensure all dependencies are installed correctly.
   - If vector store features (e.g., FAISS) are not needed, they can be omitted from the `app.py`.

2. **Docker Setup:**
   - Ensure Docker and Docker Compose are installed and properly configured.
   - If issues occur, try stopping and removing the containers and volumes:
     ```bash
     docker-compose down -v
     ```

3. **Widoco Documentation:**
   - Make sure the ontology file (`wirr_project3.rdf`) is present in the correct folder.
   - Run the script with the required permissions (you may need to use `chmod +x widoco.sh`).

---

This setup provides a scalable environment for ontology-based query systems and documentation generation, making it easier to query, explore, and document complex ontologies.