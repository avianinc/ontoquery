# app.py

import streamlit as st
import rdflib
from rdflib.plugins.sparql import prepareQuery
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Set cache directory paths
os.environ['TRANSFORMERS_CACHE'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['HF_DATASETS_CACHE'] = '/app/cache'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Constants
OLLAMA_SERVER_URL = "http://192.168.86.100:11434"  # Adjust to your LLM endpoint
FUSEKI_DATA_ENDPOINT = "http://fuseki:3030/ds/data"
FUSEKI_QUERY_ENDPOINT = "http://fuseki:3030/ds/query"
FUSEKI_UPDATE_ENDPOINT = "http://fuseki:3030/ds/update"
USERNAME = "admin"
PASSWORD = "adminpassword"

# Initialize the embedding model with GPU support
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda', cache_folder='/app/cache')

# Define standard prefixes
STANDARD_PREFIXES = {
    'rdf': str(rdflib.RDF),
    'rdfs': str(rdflib.RDFS),
    'owl': str(rdflib.OWL),
    'xsd': str(rdflib.XSD),
    'foaf': str(rdflib.namespace.FOAF),
    'dc': str(rdflib.namespace.DC),
    'dcterms': str(rdflib.namespace.DCTERMS),
    'skos': str(rdflib.namespace.SKOS),
    'prov': 'http://www.w3.org/ns/prov#',
    'vann': 'http://purl.org/vocab/vann/',
    'void': 'http://rdfs.org/ns/void#',
    'schema': 'http://schema.org/',
    'sh': 'http://www.w3.org/ns/shacl#',
    'geo': 'http://www.opengis.net/ont/geosparql#',
    'time': 'http://www.w3.org/2006/time#',
    'sosa': 'http://www.w3.org/ns/sosa/',
    'ssn': 'http://www.w3.org/ns/ssn/',
    'doap': 'http://usefulinc.com/ns/doap#',
    'odrl': 'http://www.w3.org/ns/odrl/2/',
    'org': 'http://www.w3.org/ns/org#',
    'qb': 'http://purl.org/linked-data/cube#',
    'dcam': 'http://purl.org/dc/dcam/',
    'dcmitype': 'http://purl.org/dc/dcmitype/',
    'wgs': 'http://www.w3.org/2003/01/geo/wgs84_pos#',
    'xml': 'http://www.w3.org/XML/1998/namespace',
    'csvw': 'http://www.w3.org/ns/csvw#',
    'brick': 'https://brickschema.org/schema/Brick#',
    'dcat': 'http://www.w3.org/ns/dcat#',
    'prof': 'http://www.w3.org/ns/dx/prof/',
    'xhtml': 'http://www.w3.org/1999/xhtml/vocab#',
    # Add any other standard prefixes you need
}

def extract_prefixes(file_content, file_format):
    """
    Extract custom prefixes and namespaces from the ontology file content.
    """
    graph = rdflib.Graph()
    graph.parse(data=file_content, format=file_format)

    # Define standard prefixes to exclude
    standard_prefixes = set(STANDARD_PREFIXES.keys())

    # Extract namespaces
    namespaces = {}
    for prefix, namespace in graph.namespaces():
        if prefix not in standard_prefixes:
            namespace_str = str(namespace)
            # Ensure namespace ends with '#' or '/' based on URI
            if not namespace_str.endswith(('#', '/')):
                if '#' in namespace_str:
                    namespace_str += '#'
                else:
                    namespace_str += '/'
            namespaces[prefix] = namespace_str

    # Ensure that at least one custom prefix is extracted
    if not namespaces:
        st.error("No custom prefixes found in the ontology.")
        raise Exception("No custom prefixes found in the ontology.")

    return namespaces

def extract_ontology_elements(file_content, file_format, prefixes):
    """
    Extract classes, object properties, data properties, and individuals from the ontology.
    """
    graph = rdflib.Graph()
    graph.parse(data=file_content, format=file_format)

    # Extract classes
    classes = set()
    for s in graph.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        classes.add(s)

    # Extract object properties
    object_properties = set()
    for s in graph.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        object_properties.add(s)

    # Extract data properties
    data_properties = set()
    for s in graph.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty):
        data_properties.add(s)

    # Extract individuals
    individuals = set()
    for s in graph.subjects(rdflib.RDF.type, None):
        if (s, rdflib.RDF.type, rdflib.OWL.Class) not in graph and \
           (s, rdflib.RDF.type, rdflib.OWL.ObjectProperty) not in graph and \
           (s, rdflib.RDF.type, rdflib.OWL.DatatypeProperty) not in graph:
            individuals.add(s)

    return classes, object_properties, data_properties, individuals, graph

def get_label(uri):
    """
    Extract the label from the URI.
    """
    return uri.split('#')[-1] if '#' in uri else uri.rsplit('/', 1)[-1]

def generate_embeddings(ontology_elements):
    """
    Generate embeddings for ontology elements.
    """
    labels = [get_label(str(e)) for e in ontology_elements]
    embeddings = embedding_model.encode(labels, convert_to_tensor=True, show_progress_bar=False)
    embeddings = embeddings.detach().cpu().numpy()
    return labels, embeddings

def build_vector_database(embeddings):
    """
    Build a FAISS GPU vector database from embeddings.
    """
    # Convert embeddings to float32 if not already
    embeddings = embeddings.astype('float32')

    # Initialize GPU resources
    res = faiss.StandardGpuResources()

    # Define the index
    dimension = embeddings.shape[1]
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0  # GPU device id

    # Build the index on GPU
    gpu_index = faiss.GpuIndexFlatL2(res, dimension, flat_config)
    gpu_index.add(embeddings)
    return gpu_index

def search_relevant_elements(question, labels, index, ontology_elements, k=20):
    """
    Search for relevant ontology elements based on the question.
    """
    question_embedding = embedding_model.encode([question], convert_to_tensor=True, show_progress_bar=False)
    question_embedding = question_embedding.detach().cpu().numpy().astype('float32')

    distances, indices = index.search(question_embedding, k)
    relevant_elements = [ontology_elements[idx] for idx in indices[0]]
    return relevant_elements

def create_ontology_summary_from_elements(relevant_elements, graph, prefixes):
    """
    Create a concise summary of the relevant ontology elements to include in the prompt.
    """
    def shorten_uri(uri):
        for prefix, namespace in prefixes.items():
            if uri.startswith(namespace):
                local_part = uri.replace(namespace, '')
                return f"{prefix}:{local_part}"
        # Handle standard namespaces
        for prefix, namespace in STANDARD_PREFIXES.items():
            if uri.startswith(namespace):
                local_part = uri.replace(namespace, '')
                return f"{prefix}:{local_part}"
        return uri

    classes_short = []
    object_props_short = []
    data_props_short = []
    individuals_short = []

    for element in relevant_elements:
        uri = str(element)
        short_uri = shorten_uri(uri)
        if (element, rdflib.RDF.type, rdflib.OWL.Class) in graph:
            classes_short.append(short_uri)
        elif (element, rdflib.RDF.type, rdflib.OWL.ObjectProperty) in graph:
            object_props_short.append(short_uri)
        elif (element, rdflib.RDF.type, rdflib.OWL.DatatypeProperty) in graph:
            data_props_short.append(short_uri)
        else:
            individuals_short.append(short_uri)

    summary = f"""
Classes:
{', '.join(classes_short)}

Object Properties:
{', '.join(object_props_short)}

Data Properties:
{', '.join(data_props_short)}

Individuals:
{', '.join(individuals_short)}
"""
    return summary.strip()

def generate_sparql_query(question, prefixes, ontology_summary):
    """
    Generate a SPARQL query from a natural language question using the LLM.
    """
    # Construct the PREFIX declarations
    prefix_declarations = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in prefixes.items()])
    standard_prefixes = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in STANDARD_PREFIXES.items()])

    # Combine all prefixes
    all_prefixes = f"{standard_prefixes}\n{prefix_declarations}"

    # Create the prompt with the ontology summary and explicit instructions
    prompt = f"""
You are an expert in SPARQL and ontologies.

Given the following prefixes:

{all_prefixes}

Ontology Elements:
{ontology_summary}

Convert the following natural language question into a SPARQL query, using only the prefixes, classes, properties, and individuals provided above.

Ensure that:
- You include all necessary PREFIX declarations exactly as provided above in your output.
- Do not introduce any new prefixes or assume any default prefixes.
- Use only the prefixes as defined above.
- The query is syntactically correct and complete.
- Provide only the SPARQL query. Do not include any explanations or comments.

Question: "{question}"

SPARQL Query:
"""
    # Prepare the request payload for the LLM API
    payload = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False,
    }

    headers = {'Content-Type': 'application/json'}

    response = requests.post(
        f"{OLLAMA_SERVER_URL}/api/generate",
        data=json.dumps(payload),
        headers=headers
    )

    if response.status_code != 200:
        raise Exception(f"LLM API call failed: {response.text}")

    # Extract the generated text
    result = response.json()
    sparql_query = result.get('response', '')

    if not any(keyword in sparql_query.lower() for keyword in ['select', 'ask', 'construct', 'describe']):
        raise Exception("The LLM did not generate a valid SPARQL query. Please try rephrasing your question.")

    return sparql_query.strip()

def post_process_sparql_query(sparql_query, prefixes):
    """
    Post-process the SPARQL query to fix prefixes and namespaces, and remove any extra text after the last closing brace.
    """
    # Combine custom and standard prefixes
    all_prefixes = {**STANDARD_PREFIXES, **prefixes}

    # Parse existing PREFIX declarations in the generated query
    prefix_pattern = re.compile(r'PREFIX\s+(\w+):\s*<([^>]+)>', re.IGNORECASE)
    lines = sparql_query.strip().split('\n')
    query_lines = []
    for line in lines:
        match = prefix_pattern.match(line.strip())
        if match:
            prefix, uri = match.groups()
            # Replace with correct URI if prefix is known
            if prefix in all_prefixes:
                correct_uri = all_prefixes[prefix]
                line = f"PREFIX {prefix}: <{correct_uri}>"
            else:
                # Remove unknown prefixes
                continue
        query_lines.append(line)

    # Remove any text after the last closing brace '}'
    query_body = '\n'.join(query_lines)
    last_closing_brace_index = query_body.rfind('}')
    if last_closing_brace_index != -1:
        query_body = query_body[:last_closing_brace_index+1]  # Include the closing brace

    # Reconstruct the PREFIX declarations
    prefix_declarations = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in all_prefixes.items()])
    # Ensure 'rdf' prefix is included
    if 'rdf' not in prefixes:
        prefix_declarations = f"PREFIX rdf: <{STANDARD_PREFIXES['rdf']}>\n{prefix_declarations}"

    # Combine the correct PREFIX declarations with the query body
    corrected_query = f"{prefix_declarations}\n\n{query_body}"

    return corrected_query.strip()

def execute_sparql_query(sparql_query):
    """
    Execute the SPARQL query against the Fuseki server.
    """
    headers = {'Content-Type': 'application/sparql-query', 'Accept': 'application/sparql-results+json'}

    response = requests.post(
        FUSEKI_QUERY_ENDPOINT,
        data=sparql_query.encode('utf-8'),
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )

    if response.status_code != 200:
        raise Exception(f"SPARQL query execution failed: {response.text}")

    results = response.json()
    return results

def load_ontology_to_fuseki(file_content, file_format):
    """
    Load ontology into the Fuseki server.
    """
    if file_format == 'xml':
        content_type = 'application/rdf+xml'
    elif file_format == 'turtle':
        content_type = 'text/turtle'
    else:
        content_type = 'application/octet-stream'  # default

    headers = {'Content-Type': content_type}

    # Clear existing data
    clear_response = requests.post(
        FUSEKI_UPDATE_ENDPOINT,
        data="DROP ALL".encode('utf-8'),
        headers={'Content-Type': 'application/sparql-update'},
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )

    if clear_response.status_code not in [200, 204]:
        raise Exception(f"Failed to clear Fuseki dataset: Status Code: {clear_response.status_code}, Reason: {clear_response.reason}, Response Text: {clear_response.text}")

    # Load new data
    response = requests.post(
        FUSEKI_DATA_ENDPOINT,
        data=file_content.encode('utf-8'),
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )

    if response.status_code not in (200, 201):
        raise Exception(f"Failed to load ontology into Fuseki: Status Code: {response.status_code}, Reason: {response.reason}, Response Text: {response.text}")

def detect_file_format(file_name):
    """
    Detect the format of the ontology file based on its extension.
    """
    if file_name.endswith('.ttl'):
        return 'turtle'
    elif file_name.endswith('.rdf') or file_name.endswith('.xml'):
        return 'xml'
    elif file_name.endswith('.owl'):
        return 'xml'
    else:
        return 'turtle'  # Default format

def display_results(results):
    """
    Display SPARQL query results in Streamlit.
    """
    if 'results' in results:
        headers = results['head']['vars']
        rows = results['results']['bindings']

        data = []
        for row in rows:
            data.append([row.get(var, {}).get('value', '') for var in headers])

        df = pd.DataFrame(data, columns=headers)
        st.dataframe(df)
    else:
        st.write("No results found.")

def main():
    st.title("Ontology Question Answering System")

    # Ontology Upload
    uploaded_file = st.file_uploader("Upload your ontology file", type=['owl', 'rdf', 'ttl'])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode('utf-8')
        file_format = detect_file_format(uploaded_file.name)

        try:
            # Extract prefixes
            prefixes = extract_prefixes(file_content, file_format)
            st.subheader("Extracted Prefixes and Namespaces")
            st.code('\n'.join([f"{k}: {v}" for k, v in prefixes.items()]))

            # Extract ontology elements
            classes, object_properties, data_properties, individuals, graph = extract_ontology_elements(file_content, file_format, prefixes)
            ontology_elements = list(classes) + list(object_properties) + list(data_properties) + list(individuals)

            # Generate embeddings for ontology elements
            labels, embeddings = generate_embeddings(ontology_elements)

            # Build the vector database
            index = build_vector_database(embeddings)

            # Load into Fuseki
            with st.spinner("Loading ontology into Fuseki..."):
                load_ontology_to_fuseki(file_content, file_format)
            st.success("Ontology loaded into Fuseki successfully.")

            # Question Input
            question = st.text_input("Ask a question about the ontology:")
            if question:
                # Search for relevant ontology elements
                relevant_elements = search_relevant_elements(question, labels, index, ontology_elements)

                # Create ontology summary from relevant elements
                ontology_summary = create_ontology_summary_from_elements(relevant_elements, graph, prefixes)
                st.subheader("Ontology Summary")
                st.text(ontology_summary)

                # Generate SPARQL query
                with st.spinner("Generating SPARQL query..."):
                    sparql_query = generate_sparql_query(question, prefixes, ontology_summary)
                    st.subheader("Generated SPARQL Query (Raw Output)")
                    st.code(sparql_query, language='sparql')

                    # Post-process the SPARQL query
                    sparql_query = post_process_sparql_query(sparql_query, prefixes)

                st.subheader("Processed SPARQL Query")
                st.code(sparql_query, language='sparql')

                # Validate SPARQL syntax
                try:
                    prepareQuery(sparql_query)
                except Exception as e:
                    st.error(f"Invalid SPARQL query syntax: {str(e)}")
                    return

                # Execute SPARQL query
                try:
                    with st.spinner("Executing SPARQL query..."):
                        results = execute_sparql_query(sparql_query)
                    st.subheader("Query Results")
                    display_results(results)
                except Exception as e:
                    st.error(f"Failed to execute SPARQL query: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
