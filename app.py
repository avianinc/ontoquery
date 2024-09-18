import os
import streamlit as st
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
import json
import rdflib

# Set the page layout to wide
st.set_page_config(layout="wide")

# Define the Ollama server URL
OLLAMA_SERVER_URL = "http://192.168.86.100:11434"

# Define Fuseki endpoints
FUSEKI_UPDATE_ENDPOINT = "http://fuseki:3030/ds/update"
FUSEKI_DATA_ENDPOINT = "http://fuseki:3030/ds/data"
FUSEKI_QUERY_ENDPOINT = "http://fuseki:3030/ds/sparql"
FUSEKI_SERVER_URL = "http://localhost:3030"
WIDOCO_DOC_URL = "http://localhost:8080/index-en.html"

# Directory to store ontology files
ONTOLOGY_FOLDER = "data/ontologies"
QUERY_FOLDER = "data/queries"

# Ensure the ontology and query folders exist
os.makedirs(ONTOLOGY_FOLDER, exist_ok=True)
os.makedirs(QUERY_FOLDER, exist_ok=True)

# Initialize session state
if "generated_query" not in st.session_state:
    st.session_state.generated_query = ""
if "ontology_structure" not in st.session_state:
    st.session_state.ontology_structure = {}
if "prefixes" not in st.session_state:
    st.session_state.prefixes = {}
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

# ------------------- Sidebar Ontology Management -----------------------
st.sidebar.title("Ontology Management")

# Ontology upload section
uploaded_file = st.sidebar.file_uploader("Upload an ontology file", type=["ttl", "rdf", "owl"])
if uploaded_file and st.sidebar.button("Upload Ontology"):
    save_path = os.path.join(ONTOLOGY_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Ontology {uploaded_file.name} saved locally.")

# List existing ontologies with delete buttons
ontology_files = os.listdir(ONTOLOGY_FOLDER)
if ontology_files:
    st.sidebar.write("Uploaded Ontologies:")
    for ontology_file in ontology_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(ontology_file)
        if col2.button("x", key=ontology_file):
            os.remove(os.path.join(ONTOLOGY_FOLDER, ontology_file))
            st.sidebar.success(f"Deleted {ontology_file}")
            st.experimental_rerun()  # Refresh the app

# ------------------- SPARQL Queries to Extract Ontology Data -----------------------
def query_fuseki(sparql_query):
    """Send a SPARQL query to the Fuseki server and return the result."""
    headers = {"Accept": "application/sparql-results+json"}
    params = {'query': sparql_query}
    response = requests.get(FUSEKI_QUERY_ENDPOINT, params=params, headers=headers,
                            auth=HTTPBasicAuth("admin", "adminpassword"))  # Replace with your Fuseki credentials
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to query Fuseki: {response.status_code} - {response.text}")
        return None

def extract_prefixes(ontology_file):
    """Extract prefixes from the ontology file using rdflib."""
    g = rdflib.Graph()
    g.parse(ontology_file)

    prefixes = {}
    for prefix, uri in g.namespace_manager.namespaces():
        prefixes[prefix] = str(uri)
    return prefixes

def extract_ontology_structure(prefixes):
    """Extract classes, object properties, data properties, and individuals from Fuseki."""
    ontology_summary = {}
    prefix_declaration = "\n".join([f"PREFIX {k}: <{v}>" for k, v in prefixes.items()])

    # Query classes
    classes_query = prefix_declaration + """
    SELECT ?class WHERE { ?class a owl:Class }
    """
    classes_result = query_fuseki(classes_query)
    if classes_result:
        classes = [binding['class']['value'].split('#')[-1] for binding in classes_result['results']['bindings']]
        ontology_summary['Classes'] = classes

    # Query object properties
    object_properties_query = prefix_declaration + """
    SELECT ?property WHERE { ?property a owl:ObjectProperty }
    """
    object_properties_result = query_fuseki(object_properties_query)
    if object_properties_result:
        object_properties = [binding['property']['value'].split('#')[-1] for binding in object_properties_result['results']['bindings']]
        ontology_summary['ObjectProperties'] = object_properties

    # Query data properties
    data_properties_query = prefix_declaration + """
    SELECT ?property WHERE { ?property a owl:DatatypeProperty }
    """
    data_properties_result = query_fuseki(data_properties_query)
    if data_properties_result:
        data_properties = [binding['property']['value'].split('#')[-1] for binding in data_properties_result['results']['bindings']]
        ontology_summary['DataProperties'] = data_properties

    # Query individuals
    individuals_query = prefix_declaration + """
    SELECT ?individual WHERE { ?individual a owl:NamedIndividual }
    """
    individuals_result = query_fuseki(individuals_query)
    if individuals_result:
        individuals = [binding['individual']['value'].split('#')[-1] for binding in individuals_result['results']['bindings']]
        ontology_summary['Individuals'] = individuals

    return ontology_summary

# ------------------- Mistral Model Interaction -----------------------
def process_with_llm(ontology_summary, user_question, model_id, namespaces):
    """Pass the summarized ontology to the Mistral model along with the user question."""
    ontology_content = json.dumps(ontology_summary, indent=2)
    
    # Include typical prefixes
    typical_prefixes = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xml": "http://www.w3.org/XML/1998/namespace",
        "xsd": "http://www.w3.org/2001/XMLSchema#"
    }
    all_namespaces = {**typical_prefixes, **namespaces}
    namespace_declaration = "\n".join([f"PREFIX {k}: <{v}>" for k, v in all_namespaces.items()])

    prompt = (
        f"{namespace_declaration}\n\n"
        f"Using the following ontology structure, generate a SPARQL query to answer the question: {user_question}\n\n"
        f"Ontology Structure:\n{ontology_content}\n\n"
        "Examples of good SPARQL queries:\n"
        "1. SELECT ?subject WHERE { ?subject a :ClassName }\n"
        "2. SELECT ?property WHERE { ?subject :hasProperty ?property }\n"
    )
    
    data = {
        "model": model_id,
        "prompt": prompt
    }
    try:
        llm_response = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=data)
        llm_response.raise_for_status()  # Raise an exception for HTTP errors

        json_lines = llm_response.text.strip().splitlines()
        combined_response = ""
        for line in json_lines:
            try:
                json_data = json.loads(line)
                combined_response += json_data.get('response', '')
            except json.JSONDecodeError:
                continue  # If the line isn't valid JSON, skip it

        return combined_response.strip()  # Return the full response (expected to be just the SPARQL query)
    except requests.RequestException as e:
        st.error(f"LLM processing error: {e}")
        return "Error processing query."

# ------------------- LLM Settings and Query Generation -----------------------
st.sidebar.title("LLM Settings")

# Fetch available models from the Ollama server
if "ollama_models" not in st.session_state:
    try:
        response = requests.get(f"{OLLAMA_SERVER_URL}/v1/models")
        response.raise_for_status()
        st.session_state.ollama_models = [model["id"] for model in response.json().get("data", [])]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        st.session_state.ollama_models = ["mistral:7b"]  # Default fallback

selected_model = st.sidebar.selectbox("Select LLM Model", st.session_state.ollama_models)

# User input for generating SPARQL query
user_question = st.sidebar.text_area("Enter your question")

# Save user question to context
if user_question:
    st.session_state.user_questions.append(user_question)

# Extract ontology structure and generate SPARQL query
if st.sidebar.button("Extract Ontology Structure"):
    ontology_files = os.listdir(ONTOLOGY_FOLDER)
    if ontology_files:
        ontology_file_path = os.path.join(ONTOLOGY_FOLDER, ontology_files[0])  # Use the first file for prefix extraction
        st.session_state.prefixes = extract_prefixes(ontology_file_path)
        ontology_structure = extract_ontology_structure(st.session_state.prefixes)
        st.session_state.ontology_structure = ontology_structure
        st.sidebar.json(ontology_structure)  # Display the extracted structure
    else:
        st.error("No ontology files found to extract structure.")

if st.sidebar.button("Generate SPARQL Query"):
    if "ontology_structure" in st.session_state:
        # Call the LLM with extracted ontology structure and user question
        st.session_state.generated_query = process_with_llm(
            st.session_state.ontology_structure, 
            user_question, 
            selected_model,
            st.session_state.prefixes  # Pass namespaces to the model
        )
        st.write("Generated Query:", st.session_state.generated_query)  # Log the query for debugging
        st.success("SPARQL query generated successfully.")
    else:
        st.error("Ontology structure not yet extracted. Please extract the ontology structure first.")

# ------------------- SPARQL Query Runner -----------------------
st.title("SPARQL Query Runner")

# Display the generated SPARQL query in a text area
query = st.text_area("Generated SPARQL Query", value=st.session_state.generated_query)

# Execute the SPARQL query
if st.button("Run Query"):
    response = requests.get(
        FUSEKI_QUERY_ENDPOINT,
        params={'query': query},
        headers={"Accept": "application/sparql-results+json"}
    )
    
    if response.status_code == 200:
        results = response.json()
        
        # Parse the SPARQL result
        columns = results['head']['vars']
        rows = []
        for binding in results['results']['bindings']:
            row = [binding.get(col, {}).get('value', '') for col in columns]
            rows.append(row)
        
        # Create a DataFrame for easier display
        df = pd.DataFrame(rows, columns=columns)
        st.table(df)  # Display the result as a table
    else:
        st.error(f"Failed to execute query: {response.status_code} - {response.text}")

# ------------------- Save Query Management -----------------------
with st.expander("Save and Manage Queries"):
    save_name = st.text_input("Enter a name to save the query", value="new_query.sparql")
    if st.button("Save Query"):
        save_path = os.path.join(QUERY_FOLDER, save_name)
        with open(save_path, "w") as file:
            file.write(query)
        st.success(f"Query saved as {save_name}")
        st.experimental_rerun()

    saved_queries = os.listdir(QUERY_FOLDER)
    selected_saved_query = st.selectbox("Choose a saved query", [""] + saved_queries)
    
    if selected_saved_query:
        with open(os.path.join(QUERY_FOLDER, selected_saved_query), "r") as file:
            saved_query_content = file.read()
        st.text_area("Loaded Saved Query", value=saved_query_content, height=150)

# ------------------- Tools and Links Section -----------------------
with st.sidebar.expander("Tools and Links"):
    st.markdown(f"[Ontology Docs](http://localhost:8080/index-en.html)", unsafe_allow_html=True)
    st.markdown(f"[Fuseki Server](http://localhost:3030)", unsafe_allow_html=True)
    st.markdown(f"[WebProtege](http://localhost:5000)", unsafe_allow_html=True)
