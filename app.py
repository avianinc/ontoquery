import os
import streamlit as st
import requests
from requests.auth import HTTPBasicAuth

# Define Fuseki endpoints
FUSEKI_UPDATE_ENDPOINT = "http://fuseki:3030/ds/update"  # Update this if necessary
FUSEKI_DATA_ENDPOINT = "http://fuseki:3030/ds/data"
FUSEKI_QUERY_ENDPOINT = "http://fuseki:3030/ds/sparql"
OLLAMA_SERVER_URL = "http://192.168.86.100:11434"  # Adjust to your LLM endpoint

# Directory to store ontology files
ONTOLOGY_FOLDER = "data/ontologies"
os.makedirs(ONTOLOGY_FOLDER, exist_ok=True)

# Authentication credentials
USERNAME = "admin"
PASSWORD = "adminpassword"

# Function to clear the dataset on Fuseki
def clear_dataset():
    clear_url = "http://fuseki:3030/ds/update"
    query = "DROP ALL"  # This will remove all data from the dataset
    response = requests.post(clear_url, data={"update": query}, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    return response.status_code == 200

# Function to upload the file to Fuseki
def upload_to_fuseki(file):
    url = FUSEKI_UPDATE_ENDPOINT
    files = {'file': (file.name, file.getvalue())}

    # Set the appropriate Content-Type based on the file extension
    if file.name.endswith('.ttl'):
        headers = {'Content-Type': 'application/x-turtle'}
    elif file.name.endswith('.owl') or file.name.endswith('.rdf'):
        headers = {'Content-Type': 'application/rdf+xml'}
    else:
        st.error("Unsupported file type.")
        return False

    # Sending the request
    response = requests.post(url, files=files, headers=headers, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    # Debugging information
    if response.status_code != 200:
        print("Upload failed with status code:", response.status_code)
        print("Response content:", response.text)  # Print the response content for debugging
    return response.status_code == 200

# Function to perform SPARQL query
def sparql_query(query):
    response = requests.get(FUSEKI_QUERY_ENDPOINT, params={'query': query}, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    return response.json()

# Function to convert text to SPARQL using the mistral:7b model
def text_to_sparql(user_input):
    prompt = f"Convert the following question into a SPARQL query: {user_input}"
    data = {"model": "mistral:7b", "prompt": prompt}
    
    response = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=data)
    if response.status_code == 200:
        return response.json().get("generated_text", "").strip()
    return None

# Streamlit app layout
st.title("Ontology Chatbot")

uploaded_file = st.file_uploader("Upload TTL, OWL, or RDF file", type=['ttl', 'owl', 'rdf'])

if uploaded_file:
    if st.button("Upload to Fuseki"):
        # Clear the dataset first
        if clear_dataset():
            st.success("Dataset cleared successfully!")
        else:
            st.error("Failed to clear dataset.")
        
        # Now upload the new file
        if upload_to_fuseki(uploaded_file):
            st.success("File uploaded successfully!")
        else:
            st.error("Failed to upload file. Check the console for more details.")

query = st.text_input("Ask a question about the ontology:")
if query:
    # Convert user question to SPARQL query
    sparql_query_string = text_to_sparql(query)

    if sparql_query_string:
        results = sparql_query(sparql_query_string)
        st.json(results)  # Display results in JSON format
    else:
        st.warning("Could not convert question to SPARQL.")
