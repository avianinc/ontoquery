import requests

# Set up session
session = requests.Session()

# Replace with your WebProtégé login URL, username, and password
login_url = 'http://192.168.86.100:5000/#login'
username = 'jdehart'
password = 'ctv2123!1'

# Login payload (this may vary depending on your WebProtégé login form)
login_payload = {
    'username': username,
    'password': password
}

# Log in to WebProtégé
login_response = session.post(login_url, data=login_payload)

# Check if login is successful
if login_response.status_code == 200:
    print("Logged in successfully.")
    
    # You can print session cookies to check if they were set correctly
    print("Cookies set after login:", session.cookies.get_dict())

    # Download the ontology using the session
    download_url = 'http://192.168.86.100:5000/download?project=976e7ca3-62b6-4f33-b73a-7c5b1414a487&revision=9223372036854775807&format=owl'
    
    # Simulate a browser request by passing headers (if needed)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = session.get(download_url, headers=headers)
    
    if response.status_code == 200:
        # Save the downloaded ontology to a file
        with open("ontology.owl", "wb") as f:
            f.write(response.content)
        print("Ontology downloaded successfully.")
    else:
        print(f"Failed to download ontology. Status code: {response.status_code}")
else:
    print(f"Failed to log in. Status code: {login_response.status_code}")





#http://192.168.86.100:5000//download?project=%22%22&revision=%22%22&format=owl
#http://192.168.86.100:5000//download?project=976e7ca3-62b6-4f33-b73a-7c5b1414a487&revision=9223372036854775807&format=owl