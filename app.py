import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import json
import rdflib  # For parsing RDF and extracting prefixes
import re      # For regex operations

# Constants
OLLAMA_SERVER_URL = "http://192.168.86.100:11434"  # Adjust to your LLM endpoint
FUSEKI_DATA_ENDPOINT = "http://fuseki:3030/ds/data"
FUSEKI_UPDATE_ENDPOINT = "http://fuseki:3030/ds/update"
USERNAME = "admin"
PASSWORD = "adminpassword"

# Function to clear the Fuseki dataset
def clear_dataset():
    clear_url = FUSEKI_UPDATE_ENDPOINT
    query = "DROP ALL"
    response = requests.post(clear_url, data={"update": query}, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    return response.status_code == 200

# Function to upload the ontology to Fuseki
def upload_to_fuseki(file):
    content_type = "application/x-turtle" if file.name.endswith('.ttl') else "application/rdf+xml"

    upload_response = requests.post(
        FUSEKI_DATA_ENDPOINT,
        headers={"Content-Type": content_type},
        data=file.getvalue(),
        auth=HTTPBasicAuth(USERNAME, PASSWORD)
    )

    return upload_response.status_code == 200

# Updated function to extract the base namespace from the uploaded ontology
def extract_base_namespace(file_content):
    g = rdflib.Graph()
    try:
        # Try parsing the ontology file in multiple formats
        for fmt in ['xml', 'turtle', 'n3', 'nt', 'json-ld']:
            try:
                g.parse(data=file_content, format=fmt)
                break
            except Exception:
                continue
        else:
            st.error("Could not parse the ontology file in any known format.")
            return None
    except Exception as e:
        st.error(f"Error parsing the ontology file: {str(e)}")
        return None

    # Attempt to extract the base namespace
    base_namespace = None
    for prefix, uri in g.namespaces():
        if str(prefix) == '':
            base_namespace = str(uri)
            break
    if base_namespace is None:
        # If no default namespace, use the first custom namespace
        for prefix, uri in g.namespaces():
            if prefix not in ['rdf', 'rdfs', 'xsd', 'owl', 'xml', '']:
                base_namespace = str(uri)
                break

    if base_namespace is None:
        st.error("Could not extract base namespace from the ontology.")
        return None

    # Ensure the base namespace ends with '#'
    if not base_namespace.endswith('#'):
        base_namespace += '#'

    return base_namespace

# Function to create the prefix string
def create_prefix_string(base_namespace):
    prefix_string = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <{base_namespace}>"""
    return prefix_string

# Function to replace unknown prefixes with 'ex:'
def replace_unknown_prefixes(sparql_query):
    # Replace any prefixes other than 'rdf:' and 'ex:' with 'ex:', but not inside URIs
    prefix_pattern = re.compile(r'(?<!<)(\b\w+):')
    all_prefixes_in_query = set(prefix_pattern.findall(sparql_query))
    allowed_prefixes = {'rdf', 'ex'}
    for prefix in all_prefixes_in_query:
        if prefix not in allowed_prefixes:
            # Replace prefixes not inside angle brackets
            sparql_query = re.sub(r'(?<!<)\b' + re.escape(prefix) + ':', 'ex:', sparql_query)
    return sparql_query

# Function to convert text to SPARQL using the LLM
def text_to_sparql(user_input, base_namespace):
    # Create the prefix string
    prefix_string = create_prefix_string(base_namespace)

    # Prepare sample queries
    sample_queries = [
        {
            'question': 'List all classes in the ontology.',
            'query': '''
SELECT DISTINCT ?class
WHERE {
  ?class rdf:type rdf:Class .
}
'''
        },
        {
            'question': 'What are the properties and values of the individual Pilot?',
            'query': '''
SELECT ?property ?value
WHERE {
  ex:Pilot ?property ?value .
}
'''
        },
        {
            'question': 'List the mass of the air vehicles.',
            'query': '''
SELECT ?airVehicle ?mass
WHERE {
  ?airVehicle rdf:type ex:AirVehicle .
  ?airVehicle ex:mass ?mass .
}
'''
        },
    ]

    # Prepare examples
    examples = ""
    for idx, sample in enumerate(sample_queries, start=1):
        examples += f"""
Example {idx}:
Question: "{sample['question']}"
SPARQL Query:

{prefix_string}

{sample['query']}

---
"""

    # Construct the prompt
    prompt = (
        "You are a helpful assistant that converts questions into SPARQL queries.\n"
        "Use 'ex:' as the prefix for the ontology namespace and 'rdf:' for RDF syntax.\n"
        "Do not use any other prefixes.\n"
        f"{examples}"
        "Now, please generate the SPARQL query for the following question:\n"
        f"Question: \"{user_input}\"\n"
        "SPARQL Query:\n\n"
        f"{prefix_string}\n\n"
    )

    # Send prompt to the LLM
    data = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=data)
    if response.status_code == 200:
        try:
            response_content = response.json()
            sparql_query_body = response_content.get("response", "").strip()

            # Combine the prefix string and the query body
            full_sparql_query = f"{prefix_string}\n\n{sparql_query_body}"

            # Replace any unknown prefixes with 'ex:'
            full_sparql_query = replace_unknown_prefixes(full_sparql_query)

            # Enclose the query within code fences
            full_sparql_query = f"```sparql\n{full_sparql_query}\n```"
            return full_sparql_query
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {e}")
            return None
    else:
        st.error("Error in response from LLM.")
        return None

# Streamlit app layout
st.title("Ontology Chatbot")

uploaded_file = st.file_uploader("Upload TTL, OWL, or RDF file", type=['ttl', 'owl', 'rdf'])

if uploaded_file:
    file_content = uploaded_file.getvalue().decode('utf-8')

    # Expander for uploaded file content
    with st.expander("Uploaded File Content", expanded=False):
        st.code(file_content)

    if st.button("Upload to Fuseki"):
        if clear_dataset():
            st.success("Dataset cleared successfully!")
        else:
            st.error("Failed to clear dataset.")

        if upload_to_fuseki(uploaded_file):
            st.success(f"Ontology {uploaded_file.name} uploaded to Fuseki.")
            # Extract base namespace
            base_namespace = extract_base_namespace(file_content)
            if base_namespace:
                with st.expander("Extracted Base Namespace", expanded=False):
                    st.write(f"Base Namespace: {base_namespace}")
        else:
            st.error("Failed to upload ontology.")

query = st.text_input("Ask a question about the ontology:")
if query:
    # Convert user question to SPARQL query
    if 'base_namespace' in locals():
        sparql_query_string = text_to_sparql(query, base_namespace)

        if sparql_query_string:
            with st.expander("Generated SPARQL Query", expanded=True):
                st.code(sparql_query_string, language='sparql')
        else:
            st.warning("Failed to generate SPARQL query.")
    else:
        st.warning("No ontology uploaded yet or base namespace not extracted.")
