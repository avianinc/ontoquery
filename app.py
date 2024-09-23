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

# Set the page configuration
st.set_page_config(layout="wide")

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

# Memory for conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

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
}

def extract_prefixes(file_content, file_format):
    graph = rdflib.Graph()
    graph.parse(data=file_content, format=file_format)

    standard_prefixes = set(STANDARD_PREFIXES.keys())
    namespaces = {}
    for prefix, namespace in graph.namespaces():
        if prefix not in standard_prefixes:
            namespace_str = str(namespace)
            if not namespace_str.endswith(('#', '/')):
                namespace_str += '#' if '#' in namespace_str else '/'
            namespaces[prefix] = namespace_str

    if not namespaces:
        st.error("No custom prefixes found in the ontology.")
        raise Exception("No custom prefixes found in the ontology.")

    return namespaces

def extract_ontology_elements(file_content, file_format):
    graph = rdflib.Graph()
    graph.parse(data=file_content, format=file_format)

    classes = set(graph.subjects(rdflib.RDF.type, rdflib.OWL.Class))
    object_properties = set(graph.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    data_properties = set(graph.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty))
    individuals = set(s for s in graph.subjects(rdflib.RDF.type, None) if
                      (s, rdflib.RDF.type, rdflib.OWL.Class) not in graph and
                      (s, rdflib.RDF.type, rdflib.OWL.ObjectProperty) not in graph and
                      (s, rdflib.RDF.type, rdflib.OWL.DatatypeProperty) not in graph)

    return classes, object_properties, data_properties, individuals, graph

def get_label(uri):
    return uri.split('#')[-1] if '#' in uri else uri.rsplit('/', 1)[-1]

def generate_ontology_summary(graph, classes, object_properties, data_properties, individuals, prefixes):
    def shorten_uri(uri):
        for prefix, namespace in prefixes.items():
            if uri.startswith(namespace):
                return f"{prefix}:{uri.replace(namespace, '')}"
        for prefix, namespace in STANDARD_PREFIXES.items():
            if uri.startswith(namespace):
                return f"{prefix}:{uri.replace(namespace, '')}"
        return uri

    subclass_relationships = [(shorten_uri(str(cls)), shorten_uri(str(sup))) for cls in classes for sup in graph.objects(cls, rdflib.RDFS.subClassOf)]
    
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

    individual_relationships = [(shorten_uri(str(subj)), shorten_uri(str(pred)), shorten_uri(str(obj))) for subj in individuals for pred, obj in graph.predicate_objects(subj)]

    summary = "### Ontology Summary\n\n"
    
    summary += "#### Classes and Subclass Relationships:\n"
    for cls in classes:
        cls_short = shorten_uri(str(cls))
        subclasses = [sc for sc, sup in subclass_relationships if sup == cls_short]
        summary += f"- **{cls_short}** is a superclass of: {', '.join(subclasses) if subclasses else 'None'}\n"

    summary += "\n#### Properties:\n"
    for prop in property_info:
        domains = ', '.join(prop["domain"])
        ranges = ', '.join(prop["range"])
        summary += f"- **{prop['property']}** ({prop['type']}):\n  - Domain: {domains}\n  - Range: {ranges}\n"

    summary += "\n#### Individuals and Their Relationships:\n"
    for ind in individuals:
        ind_short = shorten_uri(str(ind))
        relations = [f"{pred} {obj}" for subj, pred, obj in individual_relationships if subj == ind_short]
        summary += f"- **{ind_short}** is connected to:\n  - {', '.join(relations) if relations else 'None'}\n"

    return summary.strip()

def generate_sparql_query(question, prefixes, ontology_summary):
    prefix_declarations = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in {**STANDARD_PREFIXES, **prefixes}.items()])
    prompt = f"""
You are an expert in SPARQL and ontologies.

Given the following prefixes:

{prefix_declarations}

Ontology Summary:
{ontology_summary}

Convert the following natural language question into a SPARQL query:

Question: "{question}"

SPARQL Query:
"""
    payload = {"model": "mistral:7b", "prompt": prompt, "stream": False}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", data=json.dumps(payload), headers=headers, timeout=60)
        response.raise_for_status()
        sparql_query = response.json().get('response', '')

        if not any(keyword in sparql_query.lower() for keyword in ['select', 'ask', 'construct', 'describe']):
            raise Exception("Invalid SPARQL query generated.")
        return sparql_query.strip()

    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API call failed: {str(e)}")

def post_process_sparql_query(sparql_query, prefixes):
    all_prefixes = {**STANDARD_PREFIXES, **prefixes}
    lines = sparql_query.strip().split('\n')
    query_lines = []
    seen_prefixes = set()

    for line in lines:
        match = re.match(r'PREFIX\s+(\w+):\s*<([^>]+)>', line.strip())
        if match:
            prefix, uri = match.groups()
            if prefix in all_prefixes and prefix not in seen_prefixes:
                query_lines.append(f"PREFIX {prefix}: <{all_prefixes[prefix]}>")
                seen_prefixes.add(prefix)
        else:
            query_lines.append(line)

    query_body = '\n'.join(query_lines)
    last_closing_brace_index = query_body.rfind('}')
    if last_closing_brace_index != -1:
        query_body = query_body[:last_closing_brace_index + 1]

    prefix_declarations = '\n'.join([f"PREFIX {k}: <{v}>" for k, v in all_prefixes.items() if k in seen_prefixes])
    corrected_query = f"{prefix_declarations}\n\n{query_body}"
    return corrected_query.strip()

def execute_sparql_query(sparql_query):
    headers = {'Content-Type': 'application/sparql-query', 'Accept': 'application/sparql-results+json'}

    try:
        response = requests.post(FUSEKI_QUERY_ENDPOINT, data=sparql_query.encode('utf-8'), headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=60)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"SPARQL query execution failed: {str(e)}")

def load_ontology_to_fuseki(file_content, file_format):
    content_type = 'application/rdf+xml' if file_format == 'xml' else 'text/turtle'
    headers = {'Content-Type': content_type}

    requests.post(FUSEKI_UPDATE_ENDPOINT, data="DROP ALL".encode('utf-8'), headers={'Content-Type': 'application/sparql-update'}, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=60)

    response = requests.post(FUSEKI_DATA_ENDPOINT, data=file_content.encode('utf-8'), headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=120)
    if response.status_code not in (200, 201):
        raise Exception(f"Failed to load ontology into Fuseki: Status Code: {response.status_code}, Reason: {response.reason}")

def detect_file_format(file_name):
    if file_name.endswith('.ttl'):
        return 'turtle'
    elif file_name.endswith('.rdf') or file_name.endswith('.xml'):
        return 'xml'
    elif file_name.endswith('.owl'):
        return 'xml'
    else:
        return 'turtle'

def display_results(results):
    if 'results' in results:
        headers = results['head']['vars']
        rows = results['results']['bindings']
        data = [[row.get(var, {}).get('value', '').split('#')[-1] for var in headers] for row in rows]
        df = pd.DataFrame(data, columns=headers)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No results found.")

def display_gpu_utilization():
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

def display_conversation():
    if st.session_state.conversation:
        for entry in st.session_state.conversation:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**System:** {entry['system']}")
            st.write("---")

def main():
    st.title("Ontology Question Answering System")

    with st.expander("GPU Information"):
        st.write("This application uses a GPU for running the Language Model.\n The following GPU information is available:")
        display_gpu_utilization()

    # Ontology Upload
    uploaded_file = st.file_uploader("Upload your ontology file", type=['owl', 'rdf', 'ttl'])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode('utf-8')
        file_format = detect_file_format(uploaded_file.name)

        try:
            with st.expander("Ontology File Content and Extracted Information"):
                prefixes = extract_prefixes(file_content, file_format)
                st.subheader("Extracted Prefixes and Namespaces")
                st.code('\n'.join([f"{k}: {v}" for k, v in prefixes.items()]))

                classes, object_properties, data_properties, individuals, graph = extract_ontology_elements(file_content, file_format)
                ontology_summary = generate_ontology_summary(graph, classes, object_properties, data_properties, individuals, prefixes)
                st.subheader("Ontology Summary")
                st.text(ontology_summary)

            with st.spinner("Loading ontology into Fuseki..."):
                load_ontology_to_fuseki(file_content, file_format)
            st.success("Ontology loaded into Fuseki successfully.")

            # Display conversation history
            display_conversation()

            # Question Input
            question = st.text_area("Ask a question about the ontology:", height=100)
            if question:
                # Generate SPARQL query
                with st.expander("SPARQL Query Generation"):
                    with st.spinner("Generating SPARQL query..."):
                        sparql_query = generate_sparql_query(question, prefixes, ontology_summary)
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

                    # Store the question and answer in conversation memory
                    st.session_state.conversation.append({"user": question, "system": results})
                except Exception as e:
                    st.error(f"Failed to execute SPARQL query: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
