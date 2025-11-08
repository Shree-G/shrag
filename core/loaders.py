import os
import shutil
import requests
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, GitLoader
from langchain_core.documents import Document
from app.settings import GITHUB_USERNAME  # Import config from your settings file

# --- Configuration ---

# A temporary directory to clone repos into.
# This will be created and deleted during the ingestion process.
GIT_TEMP_CLONE_DIR = "./temp_git_repos/"

# --- Public Functions ---

def load_all_documents() -> List[Document]:
    """
    The main high-level function to load all documents from all sources.
    This is what your ingest.py script will call.
    """
    print("Starting document loading process...")
    
    # 1. Load PDF resume
    # We get the path from an environment variable for flexibility
    resume_path = os.getenv("RESUME_FILE_PATH", "data/Your_Resume.pdf")
    resume_docs = load_resume(resume_path)
    
    # 2. Load all GitHub READMEs
    github_docs = load_all_github_readmes(username=GITHUB_USERNAME, 
                                          clone_dir=GIT_TEMP_CLONE_DIR)
    
    all_docs = resume_docs + github_docs
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def load_resume(file_path: str) -> List[Document]:
    """
    Loads a single PDF file and returns its documents.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Resume file not found at {file_path}. Skipping.")
        return []
        
    print(f"Loading resume from: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        # Add a custom metadata tag to identify the source
        for doc in documents:
            doc.metadata["source_type"] = "resume"
            doc.metadata["source_file"] = os.path.basename(file_path)
        return documents
    except Exception as e:
        print(f"Error loading resume PDF {file_path}: {e}")
        return []

def load_all_github_readmes(username: str, clone_dir: str) -> List[Document]:
    """
    Loads the README.md file from all public, non-forked repos 
    for a given GitHub username.
    """
    print(f"Loading all GitHub READMEs for user: {username}")
    
    # 1. Get a list of repo clone URLs from the GitHub API
    try:
        repo_urls = _get_public_repo_urls(username)
        if not repo_urls:
            print(f"No public, non-forked repositories found for {username}.")
            return []
    except Exception as e:
        print(f"Error fetching repo list from GitHub API: {e}")
        return []

    all_docs = []
    
    # 2. Clean up the temp directory if it exists
    _cleanup_temp_dir(clone_dir)
    
    # 3. Iterate and load each repo
    for url in repo_urls:
        repo_name = url.split('/')[-1]
        local_repo_path = os.path.join(clone_dir, repo_name)
        
        try:
            print(f"Loading README from: {url}")
            # This lambda function is the filter
            readme_loader = GitLoader(
                repo_path=local_repo_path,
                clone_url=url,
                file_filter=lambda file_path: file_path.lower().endswith("readme.md")
            )
            
            repo_docs = readme_loader.load()
            
            # Add custom metadata for better RAG
            for doc in repo_docs:
                doc.metadata["source_type"] = "github"
                doc.metadata["repo_name"] = repo_name
                doc.metadata["source_url"] = url
                
            all_docs.extend(repo_docs)
            
        except Exception as e:
            # This is non-fatal; we just skip this repo and continue
            print(f"Error loading repo {url}: {e}. Skipping.")
    
    # 4. Clean up the temp directory after we're done
    _cleanup_temp_dir(clone_dir)
    
    print(f"Loaded {len(all_docs)} README.md files from GitHub.")
    return all_docs

# --- Private Helper Functions ---

def _get_public_repo_urls(username: str) -> List[str]:
    """
    Uses the GitHub API to get a list of public, non-forked repo URLs.
    This does NOT require an API key, but you get higher rate limits if you use one.
    """
    api_url = f"https://api.github.com/users/{username}/repos"
    
    # We add 'per_page=100' to get the max allowed and reduce API calls
    response = requests.get(api_url, params={"type": "owner", "per_page": 100})
    response.raise_for_status()  # Raise an error on a bad response (404, 500, etc.)
    
    repos = response.json()
    
    # We filter out forks to only get *your* projects
    non_forked_urls = [
        repo["clone_url"] for repo in repos 
        if not repo["fork"] and repo["clone_url"]
    ]
    
    return non_forked_urls

def _cleanup_temp_dir(dir_path: str):
    """
    Safely and recursively deletes the temporary git clone directory.
    """
    if os.path.exists(dir_path):
        try:
            # 'shutil.rmtree' is a powerful command that deletes a whole folder
            shutil.rmtree(dir_path)
            print(f"Cleaned up temp directory: {dir_path}")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {dir_path}: {e}")