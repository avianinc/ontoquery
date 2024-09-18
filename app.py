# app.py

import os
import streamlit as st
import rdflib
from rdflib.plugins.sparql import prepareQuery
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json
import re
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_SERVER_URL = "http://192.168.86.100:11434"  # Adjust to your LLM endpoint
FUSEKI_DATA_ENDPOINT = "http://fuseki:3030/ds/data"
FUSEKI_QUERY_ENDPOINT = "http://fuseki:3030/ds/query"
FUSEKI_UPDATE_ENDPOINT = "http://fuseki:3030/ds/update"
USERNAME = "admin"
PASSWORD = "adminpassword"

# Set cache directory paths
os.environ['TRANSFORMERS_CACHE'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['HF_DATASETS_CACHE'] = '/app/cache'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

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

def extract_ontology_elements(file_content, file_format):
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

def generate_ontology_summary(graph, classes, object_properties, data_properties, individuals, prefixes):
    """
    Generate a structured summary of the ontology elements to provide context to the LLM.
    Includes class hierarchies, property domains/ranges, and relationships.
    """
    def shorten_uri(uri):
        for prefix, namespace in prefixes.items():
            if uri.startswith(namespace):
                local_part = uri.replace(namespace, '')
                return f"{prefix}:{local_part}"
        for prefix, namespace in STANDARD_PREFIXES.items():
            if uri.startswith(namespace):
                local_part = uri.replace(namespace, '')
                return f"{prefix}:{local_part}"
        return uri

    # Extract subclass relationships
    subclass_relationships = []
    for cls in classes:
        for superclass in graph.objects(cls, rdflib.RDFS.subClassOf):
            subclass_relationships.append((shorten_uri(str(cls)), shorten_uri(str(superclass))))

    # Extract property domains and ranges
    property_info = []
    for prop in object_properties.union(data_properties):
        domains = list(graph.objects(prop, rdflib.RDFS.domain))
        ranges = list(graph.objects(prop, rdflib.RDFS.range))
        domains_short = [shorten_uri(str(d)) for d in domains] if domains else ["None"]
        ranges_short = [shorten_uri(str(r)) for r in ranges] if ranges else ["None"]
        prop_type = "ObjectProperty" if prop in object_properties else "DatatypeProperty"
        property_info.append({
            "property": shorten_uri(str(prop)),
            "type": prop_type,
            "domain": domains_short,
            "range": ranges_short
        })

    # Extract relationships between individuals via object properties
    individual_relationships = []
    for subject in individuals:
        for predicate, obj in graph.predicate_objects(subject):
            if predicate not in data_properties and predicate not in object_properties:
                individual_relationships.append((shorten_uri(str(subject)), shorten_uri(str(predicate)), shorten_uri(str(obj))))

    # Generate summary
    summary = "### Ontology Summary\n\n"

    # Classes and Subclass Hierarchies
    summary += "#### Classes and Subclass Relationships:\n"
    for cls in classes:
        cls_short = shorten_uri(str(cls))
        subclasses = [sc for sc, sup in subclass_relationships if sup == cls_short]
        if subclasses:
            summary += f"- **{cls_short}** is a superclass of: {', '.join(subclasses)}\n"
        else:
            summary += f"- **{cls_short}** has no subclasses.\n"

    summary += "\n#### Properties:\n"
    for prop in property_info:
        domains = ', '.join(prop["domain"])
        ranges = ', '.join(prop["range"])
        summary += f"- **{prop['property']}** ({prop['type']}):\n"
        summary += f"  - Domain: {domains}\n"
        summary += f"  - Range: {ranges}\n"

    summary += "\n#### Individuals and Their Relationships:\n"
    for ind in individuals:
        ind_short = shorten_uri(str(ind))
        relations = [f"{pred} {obj}" for subj, pred, obj in individual_relationships if subj == ind_short]
        if relations:
            summary += f"- **{ind_short}** is connected to:\n"
            for rel in relations:
                summary += f"  - {rel}\n"
        else:
            summary += f"- **{ind_short}** has no relationships.\n"

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

Ontology Summary:
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

    try:
        response = requests.post(
            f"{OLLAMA_SERVER_URL}/api/generate",
            data=json.dumps(payload),
            headers=headers,
            timeout=60  # Increased timeout to 60 seconds
        )

        if response.status_code != 200:
            raise Exception(f"LLM API call failed: {response.status_code} {response.reason} {response.text}")

        # Extract the generated text
        result = response.json()
        sparql_query = result.get('response', '')

        if not any(keyword in sparql_query.lower() for keyword in ['select', 'ask', 'construct', 'describe']):
            raise Exception("The LLM did not generate a valid SPARQL query. Please try rephrasing your question.")

        # Debug: Display the generated query
        st.write("### Generated SPARQL Query:")
        st.code(sparql_query, language='sparql')
        logger.info(f"Generated SPARQL Query: {sparql_query}")

        return sparql_query.strip()

    except requests.exceptions.Timeout:
        raise Exception("LLM API call timed out. Please check your network connection or try again later.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API call failed: {str(e)}")

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
    seen_prefixes = set()

    for line in lines:
        match = prefix_pattern.match(line.strip())
        if match:
            prefix, uri = match.groups()
            if prefix in all_prefixes and prefix not in seen_prefixes:
                correct_uri = all_prefixes[prefix]
                query_lines.append(f"PREFIX {prefix}: <{correct_uri}>")
                seen_prefixes.add(prefix)
            else:
                # Skip unknown or duplicate prefixes
                continue
        else:
            query_lines.append(line)

    # Remove any text after the last closing brace '}'
    query_body = '\n'.join(query_lines)
    last_closing_brace_index = query_body.rfind('}')
    if last_closing_brace_index != -1:
        query_body = query_body[:last_closing_brace_index+1]  # Include the closing brace

    # Reconstruct the PREFIX declarations
    prefix_declarations = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in all_prefixes.items() if k in seen_prefixes])
    # Ensure 'rdf' prefix is included
    if 'rdf' not in seen_prefixes:
        prefix_declarations = f"PREFIX rdf: <{STANDARD_PREFIXES['rdf']}>\n{prefix_declarations}"

    # Combine the correct PREFIX declarations with the query body
    corrected_query = f"{prefix_declarations}\n\n{query_body}"

    # Debug: Display the corrected query
    st.write("### Processed SPARQL Query:")
    st.code(corrected_query, language='sparql')
    logger.info(f"Processed SPARQL Query: {corrected_query}")

    return corrected_query.strip()

def execute_sparql_query(sparql_query):
    """
    Execute the SPARQL query against the Fuseki server.
    """
    headers = {'Content-Type': 'application/sparql-query', 'Accept': 'application/sparql-results+json'}

    try:
        response = requests.post(
            FUSEKI_QUERY_ENDPOINT,
            data=sparql_query.encode('utf-8'),
            headers=headers,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            timeout=60  # Increased timeout to 60 seconds
        )

        if response.status_code != 200:
            raise Exception(f"SPARQL query execution failed: Status Code: {response.status_code}, Reason: {response.reason}, Response Text: {response.text}")

        results = response.json()
        logger.info("SPARQL query executed successfully.")
        return results

    except requests.exceptions.Timeout:
        raise Exception("SPARQL query execution timed out. Please check your network connection or try again later.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"SPARQL query execution failed: {str(e)}")

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
    logger.info("Sending DROP ALL command to Fuseki.")
    clear_response = requests.post(
        FUSEKI_UPDATE_ENDPOINT,
        data="DROP ALL".encode('utf-8'),
        headers={'Content-Type': 'application/sparql-update'},
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        timeout=60  # Increased timeout to 60 seconds
    )

    if clear_response.status_code not in [200, 204]:
        raise Exception(f"Failed to clear Fuseki dataset: Status Code: {clear_response.status_code}, Reason: {clear_response.reason}, Response Text: {clear_response.text}")
    else:
        logger.info(f"Fuseki dataset cleared successfully with status code: {clear_response.status_code}")
        st.write(f"Fuseki dataset cleared successfully with status code: {clear_response.status_code}")

    # Load new data
    logger.info("Loading new ontology data into Fuseki.")
    response = requests.post(
        FUSEKI_DATA_ENDPOINT,
        data=file_content.encode('utf-8'),
        headers=headers,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        timeout=120  # Increased timeout to 120 seconds for large ontologies
    )

    if response.status_code not in (200, 201):
        raise Exception(f"Failed to load ontology into Fuseki: Status Code: {response.status_code}, Reason: {response.reason}, Response Text: {response.text}")
    else:
        logger.info(f"Ontology loaded into Fuseki successfully with status code: {response.status_code}")
        st.write(f"Ontology loaded into Fuseki successfully with status code: {response.status_code}")

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
            data.append([row.get(var, {}).get('value', '').split('#')[-1] for var in headers])

        df = pd.DataFrame(data, columns=headers)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No results found.")

def display_gpu_utilization():
    """
    Display GPU utilization statistics.
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        with st.container(border=True):
            st.write(f"**GPU Name:** {gpu_name}")
            st.write(f"**Memory Allocated:** {memory_allocated:.2f} GB")
            st.write(f"**Memory Reserved:** {memory_reserved:.2f} GB")
    else:
        st.write("**No GPU available.**")

def main():
    st.title("Ontology Question Answering System")

    with st.expander("GPU Information"):
        st.write("This application uses a GPU for running the Language Model.\n The following GPU information is available:")
        # Display GPU Utilization
        display_gpu_utilization()

    # Ontology Upload
    uploaded_file = st.file_uploader("Upload your ontology file", type=['owl', 'rdf', 'ttl'])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode('utf-8')
        file_format = detect_file_format(uploaded_file.name)

        try:
            with st.expander("Ontology File Content and Extracted Information"):
                # Extract prefixes
                prefixes = extract_prefixes(file_content, file_format)
                st.subheader("Extracted Prefixes and Namespaces")
                st.code('\n'.join([f"{k}: {v}" for k, v in prefixes.items()]))

                # Extract ontology elements
                classes, object_properties, data_properties, individuals, graph = extract_ontology_elements(file_content, file_format)
                ontology_summary = generate_ontology_summary(graph, classes, object_properties, data_properties, individuals, prefixes)
                st.subheader("Ontology Summary")
                st.text(ontology_summary)

            # Load into Fuseki
            with st.spinner("Loading ontology into Fuseki..."):
                load_ontology_to_fuseki(file_content, file_format)
            st.success("Ontology loaded into Fuseki successfully.")

            # Question Input
            question = st.text_area("Ask a question about the ontology:", height=100)
            if question:
                # Generate SPARQL query
                with st.expander("SPARQL Query Generation"):
                    with st.spinner("Generating SPARQL query..."):
                        sparql_query = generate_sparql_query(question, prefixes, ontology_summary)

                        # Post-process the SPARQL query
                        sparql_query = post_process_sparql_query(sparql_query, prefixes)

                    # Validate SPARQL syntax
                    try:
                        prepareQuery(sparql_query)
                    except Exception as e:
                        st.error(f"Invalid SPARQL query syntax: {str(e)}")
                        return

                # Execute SPARQL query
                try:
                    with st.container(border=True):
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
