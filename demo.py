import requests
import json
import requests
from datetime import datetime
import urllib.request
import urllib.error
import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# demo setup
from openai import AzureOpenAI

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key="",  
    api_version="2024-10-21",
    azure_endpoint="https://.openai.azure.com"
)

def summarize_text(text, max_length=10):
    system_message = "You are a helpful assistant that summarizes text in markdown format."
    system_message += " Provide only the summary, without any additional explanations."
    user_input = f"Summarize this `README.md` text in about {max_length} words: ```README.md\n{text}```"
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo", depending on your preference
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
    )

    return response.choices[0].message.content.strip()
# Replace 'your_username' with the GitHub username you're interested in
# Replace 'your_personal_access_token' with your GitHub Personal Access Token
# Specify the start and end dates in 'YYYY-MM-DD' format
username = 'ranfysvalle02'
token = ''
start_date = '2024-11-14'
end_date = '2024-11-15'

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
    
def download_text_from_url(url = ""):
    if url == "":
        return ""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        print("Tokens:", len(word_tokenize(response.text)))
        return response.text
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
# Define the CriticalVectors class
class CriticalVectors:
    """
    A robust class to select the most relevant chunks from a text using various strategies,
    """

    def __init__(
        self,
        chunk_size=500,
        strategy='kmeans',
        num_clusters='auto',
        embeddings_model=None,
        split_method='sentences',
        max_tokens_per_chunk=512,
        use_faiss=False
    ):
        """
        Initializes CriticalVectors.

        Parameters:
        - chunk_size (int): Size of each text chunk in characters.
        - strategy (str): Strategy to use for selecting chunks ('kmeans', 'agglomerative').
        - num_clusters (int or 'auto'): Number of clusters (used in clustering strategies). If 'auto', automatically determine the number of clusters.
        - embeddings_model: Embedding model to use. If None, uses OllamaEmbeddings with 'nomic-embed-text' model.
        - split_method (str): Method to split text ('sentences', 'paragraphs').
        - max_tokens_per_chunk (int): Maximum number of tokens per chunk when splitting.
        - use_faiss (bool): Whether to use FAISS for clustering.
        """
        # Validate chunk_size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        self.chunk_size = chunk_size

        # Validate strategy
        valid_strategies = ['kmeans', 'agglomerative']
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}.")
        self.strategy = strategy

        # Validate num_clusters
        if num_clusters != 'auto' and (not isinstance(num_clusters, int) or num_clusters <= 0):
            raise ValueError("num_clusters must be a positive integer or 'auto'.")
        self.num_clusters = num_clusters

        # Set embeddings_model
        if embeddings_model is None:
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            self.embeddings_model = embeddings_model

        # Set splitting method and max tokens per chunk
        self.split_method = split_method
        self.max_tokens_per_chunk = max_tokens_per_chunk

        # Set FAISS usage
        self.use_faiss = use_faiss

    

    def split_text(self, text, method='sentences', max_tokens_per_chunk=512):
        """
        Splits the text into chunks based on the specified method.

        Parameters:
        - text (str): The input text to split.
        - method (str): Method to split text ('sentences', 'paragraphs').
        - max_tokens_per_chunk (int): Maximum number of tokens per chunk.

        Returns:
        - List[str]: A list of text chunks.
        """
        # Validate text
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("text must be a non-empty string.")

        if method == 'sentences':
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ''
            current_tokens = 0
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                num_tokens = len(tokens)
                if current_tokens + num_tokens <= max_tokens_per_chunk:
                    current_chunk += ' ' + sentence
                    current_tokens += num_tokens
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_tokens = num_tokens
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        elif method == 'paragraphs':
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ''
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += '\n\n' + para
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        else:
            raise ValueError("Invalid method for splitting text.")

    def compute_embeddings(self, chunks):
        """
        Computes embeddings for each chunk.

        Parameters:
        - chunks (List[str]): List of text chunks.

        Returns:
        - np.ndarray: Embeddings of the chunks.
        """
        # Validate chunks
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list of strings.")

        try:
            embeddings = self.embeddings_model.embed_documents(chunks)
            embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error computing embeddings: {e}")

    def select_chunks(self, chunks, embeddings):
        """
        Selects the most relevant chunks based on the specified strategy.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.

        Returns:
        - List[str]: Selected chunks.
        """
        num_chunks = len(chunks)
        num_clusters = self.num_clusters

        # Automatically determine number of clusters if set to 'auto'
        if num_clusters == 'auto':
            num_clusters = max(1, int(np.ceil(np.sqrt(num_chunks))))
        else:
            num_clusters = min(num_clusters, num_chunks)

        if self.strategy == 'kmeans':
            return self._select_chunks_kmeans(chunks, embeddings, num_clusters)
        elif self.strategy == 'agglomerative':
            return self._select_chunks_agglomerative(chunks, embeddings, num_clusters)
        else:
            # This should not happen due to validation in __init__
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_chunks_kmeans(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using KMeans clustering.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        if self.use_faiss:
            try:
                d = embeddings.shape[1]
                kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
                kmeans.train(embeddings)
                D, I = kmeans.index.search(embeddings, 1)
                labels = I.flatten()
            except Exception as e:
                raise RuntimeError(f"Error in FAISS KMeans clustering: {e}")
        else:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
                kmeans.fit(embeddings)
                labels = kmeans.labels_
            except Exception as e:
                raise RuntimeError(f"Error in KMeans clustering: {e}")

        # Find the closest chunk to each cluster centroid
        try:
            if self.use_faiss:
                centroids = kmeans.centroids
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                D, closest_indices = index.search(centroids, 1)
                closest_indices = closest_indices.flatten()
            else:
                from sklearn.metrics import pairwise_distances_argmin_min
                closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            selected_chunks = [chunks[idx] for idx in closest_indices]
            return selected_chunks
        except Exception as e:
            raise RuntimeError(f"Error selecting chunks: {e}")

    def _select_chunks_agglomerative(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using Agglomerative Clustering.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in Agglomerative Clustering: {e}")

        selected_indices = []
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0).astype('float32').reshape(1, -1)
            # Find the chunk closest to the centroid
            if self.use_faiss:
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(cluster_embeddings)
                D, I = index.search(centroid, 1)
                closest_index_in_cluster = I[0][0]
            else:
                from sklearn.metrics import pairwise_distances_argmin_min
                closest_index_in_cluster, _ = pairwise_distances_argmin_min(centroid, cluster_embeddings)
                closest_index_in_cluster = closest_index_in_cluster[0]
            selected_indices.append(cluster_indices[closest_index_in_cluster])

        selected_chunks = [chunks[idx] for idx in selected_indices]
        return selected_chunks

    def get_relevant_chunks(self, text):
        """
        Gets the most relevant chunks from the text.

        Parameters:
        - text (str): The input text.

        Returns:
        - List[str]: Selected chunks.
        """
        # Split the text into chunks
        chunks = self.split_text(
            text,
            method=self.split_method,
            max_tokens_per_chunk=self.max_tokens_per_chunk
        )

        if not chunks:
            return [], '', ''

        # first part
        first_part = chunks[0]
        # last part
        last_part = chunks[-1]

        # Compute embeddings for each chunk
        embeddings = self.compute_embeddings(chunks)

        # Select the most relevant chunks
        selected_chunks = self.select_chunks(chunks, embeddings)
        return selected_chunks, first_part, last_part



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



repos = get_repositories_within_date_range(username, token, start_date, end_date)
print(json.dumps(repos, indent=4))

markdown_readme_profile = f"""# README.md
[![GitHub](https://img.shields.io/badge/GitHub-{username}-blue.svg)](https://github.com/{username})


# {username}

# Repositories

"""

for repo in repos:
    print(repo)
    readme = download_readme(username, repo)
    cv = CriticalVectors()
    chunks, first_part, last_part = cv.get_relevant_chunks(readme)
    raw_text = first_part + "\n" + "\n".join(chunks) + "\n" + last_part
    summarized_txt = summarize_text(raw_text)
    
    if summarized_txt:
        print(summarized_txt)
        markdown_readme_profile += f"## repo:{repo}\n\n{summarized_txt}\n\n"
print("======================")
print(markdown_readme_profile)
print("======================")
print("END")


