import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import json
import os

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

# Function to convert text to SPARQL using the LLM
def text_to_sparql(user_input):
    prompt = f"You are a helpful assistant that converts questions into SPARQL queries.\nQuestion: {user_input}\nSPARQL Query:"
    data = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=data)

    if response.status_code == 200:
        try:
            response_content = response.json()
            return response_content.get("response", "").strip()
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
    if st.button("Upload to Fuseki"):
        if clear_dataset():
            st.success("Dataset cleared successfully!")
        else:
            st.error("Failed to clear dataset.")
        
        if upload_to_fuseki(uploaded_file):
            st.success(f"Ontology {uploaded_file.name} uploaded to Fuseki.")
        else:
            st.error("Failed to upload ontology.")

query = st.text_input("Ask a question about the ontology:")
if query:
    # Convert user question to SPARQL query
    sparql_query_string = text_to_sparql(query)

    if sparql_query_string:
        st.write("Generated SPARQL Query:")
        st.code(sparql_query_string)
    else:
        st.warning("Failed to generate SPARQL query.")
