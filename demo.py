import requests
import json
import requests
from datetime import datetime
import urllib.request
import urllib.error

def download_readme(username, repo):
    url = "https://raw.githubusercontent.com/"+username+"/"+repo+"/main/README.md"
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode()
        return data
    except urllib.error.HTTPError as e:
        return "unavailable"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def get_repositories_within_date_range(username, token, start_date, end_date):
    """
    Fetches repositories created by a GitHub user within a specified date range.

    Parameters:
    - username: GitHub username as a string.
    - token: Personal Access Token with appropriate permissions.
    - start_date: Start date in 'YYYY-MM-DD' format.
    - end_date: End date in 'YYYY-MM-DD' format.

    Returns:
    - A list of repository names created within the specified date range.
    """
    repositories = []
    page = 1
    per_page = 100  # GitHub API allows up to 100 items per page

    # Validate date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    while True:
        query = f"user:{username} created:{start_date}..{end_date}"
        url = "https://api.github.com/search/repositories"
        headers = {
            "Accept": "application/vnd.github+json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        params = {
            "q": query,
            "per_page": per_page,
            "page": page
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"GitHub API returned status code {response.status_code}: {response.json().get('message', '')}")

        data = response.json()
        if 'items' not in data or not data['items']:
            break

        repositories.extend(repo['name'] for repo in data['items'])
        page += 1

        # Break if we've fetched all available results
        if page > (data['total_count'] // per_page) + 1:
            break

    return repositories
# Replace 'your_username' with the GitHub username you're interested in
# Replace 'your_personal_access_token' with your GitHub Personal Access Token
# Specify the start and end dates in 'YYYY-MM-DD' format
username = 'ranfysvalle02'
token = ''
start_date = '2024-10-01'
end_date = '2024-12-31'

repos = get_repositories_within_date_range(username, token, start_date, end_date)
print(json.dumps(repos, indent=4))


