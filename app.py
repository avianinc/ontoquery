import os
import streamlit as st
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

# Set the page layout to wide
st.set_page_config(layout="wide")

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
if not os.path.exists(ONTOLOGY_FOLDER):
    os.makedirs(ONTOLOGY_FOLDER)

if not os.path.exists(QUERY_FOLDER):
    os.makedirs(QUERY_FOLDER)

# ------------------- Sidebar Ontology Upload and File Management -----------------------
st.sidebar.title("Ontology Management")

# Checkbox to clear existing dataset before uploading
clear_existing_data = st.sidebar.checkbox("Clear existing dataset before uploading", value=True)

# Ontology upload section
uploaded_file = st.sidebar.file_uploader("Upload an ontology file", type=["ttl", "rdf", "owl"])
if uploaded_file and st.sidebar.button("Upload Ontology"):
    file_ext = uploaded_file.name.split('.')[-1]
    content_type = "text/turtle" if file_ext == "ttl" else "application/rdf+xml"
    save_path = os.path.join(ONTOLOGY_FOLDER, uploaded_file.name)

    # Save the file locally
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Ontology {uploaded_file.name} saved locally.")

    # Clear existing Fuseki data if checkbox is selected
    if clear_existing_data:
        st.sidebar.info("Clearing existing Fuseki dataset...")
        delete_query = """
        DELETE WHERE {
            ?s ?p ?o
        }
        """
        delete_response = requests.post(
            FUSEKI_UPDATE_ENDPOINT,
            headers={"Content-Type": "application/sparql-update"},
            data=delete_query,
            auth=HTTPBasicAuth("admin", "adminpassword")  # Replace with correct credentials
        )
        if delete_response.status_code == 200:
            st.sidebar.success("Existing dataset cleared successfully.")
        else:
            st.sidebar.error(f"Failed to clear dataset: {delete_response.status_code} - {delete_response.text}")
    
    # Upload ontology to Fuseki
    upload_response = requests.post(
        FUSEKI_DATA_ENDPOINT,
        headers={"Content-Type": content_type},
        data=uploaded_file.getvalue(),
        auth=HTTPBasicAuth("admin", "adminpassword")  # Replace with correct credentials
    )
    if upload_response.status_code == 200:
        st.sidebar.success(f"Ontology {uploaded_file.name} uploaded to Fuseki.")
    else:
        st.sidebar.error(f"Failed to upload ontology: {upload_response.status_code} - {upload_response.text}")
    st.rerun()

# List existing ontologies with delete buttons
ontology_files = os.listdir(ONTOLOGY_FOLDER)
if ontology_files:
    st.sidebar.write("Uploaded Ontologies:")
    for ontology_file in ontology_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(ontology_file)
        if col2.button("X", key=ontology_file):
            os.remove(os.path.join(ONTOLOGY_FOLDER, ontology_file))
            st.sidebar.success(f"Deleted {ontology_file}")
            st.rerun()

# ------------------- SPARQL Query Management Section -----------------------
st.title("SPARQL Query Runner")

# Load existing queries
query_files = os.listdir(QUERY_FOLDER)
selected_query = st.selectbox("Choose a saved query", [""] + query_files)

# Load selected query
query = ""
if selected_query and selected_query != "":
    with open(os.path.join(QUERY_FOLDER, selected_query), "r") as file:
        query = file.read()

# Calculate text area height dynamically based on the query content
if query:
    text_length = len(query.split('\n'))  # Number of lines in the query
    height = min(500, max(100, text_length * 20)) + 50  # Dynamic height, capped at 500px
else:
    height = 100  # Default height if no query is loaded

query = st.text_area("SPARQL Query", value=query, height=height)

# Execute the Query
if st.button("Run Query"):
    response = requests.post(
        FUSEKI_QUERY_ENDPOINT,
        data=query,
        headers={
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json"
        }
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

with st.expander("Save and Manage Queries"):
    # Save, Edit, Delete Query Section
    save_name = st.text_input("Enter a name to save the query", value="new_query.sparql")
    if st.button("Save Query"):
        save_path = os.path.join(QUERY_FOLDER, save_name)
        with open(save_path, "w") as file:
            file.write(query)
        st.success(f"Query saved as {save_name}")
        st.rerun()

    if selected_query and selected_query != "" and st.button("Delete Query"):
        os.remove(os.path.join(QUERY_FOLDER, selected_query))
        st.success(f"Deleted {selected_query}")
        st.rerun()

# ------------------- Tools and Links Section -----------------------
with st.sidebar.expander("Tools and Links"):
    st.markdown(f"[Ontology Docs](http://localhost:8080/index-en.html)", unsafe_allow_html=True)
    st.markdown(f"[Fuseki Server](http://localhost:3030)", unsafe_allow_html=True)
