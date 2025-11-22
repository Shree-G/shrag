# The purpose of this file is to load all of our different sources into documents. These sources are:
# 1. Resume
# 2. GitHub Public repository README's
# FUTURE:
# 3. Transcript -> Class Information (This can probably be done with a LLM call)
# 4. Connected to my blog
# 5. Connected to my LinkedIn posts if it has information not in blogs

from langchain_community.document_loaders import PyPDFLoader, GitLoader
from langchain_core.documents import Document
import os
import requests
import shutil
from typing import List
from app.settings import Settings
from dotenv import load_dotenv

load_dotenv()

GIT_TEMP_CLONE_DIR = "./temp_git_repos/"

settings = Settings()


def load_all_documents() -> List[Document]:
    resume_file_path = os.getenv("RESUME_FILE_PATH")

    if not resume_file_path:
        print(f"Error: could not get resume file path")
        return

    resume_doc = load_resume(resume_file_path)
    readme_docs = load_github_readmes(settings.GITHUB_USERNAME, GIT_TEMP_CLONE_DIR)

    all_docs = readme_docs + resume_doc
    print(f"Total documents loaded: {len(all_docs)}")

    return all_docs


def load_resume(resumePath: str) -> List[Document]:
    print(f"Loading Resume from file path: {resumePath}")
    if not os.path.exists(resumePath):
        print(f"Error: Resume not find in file path: {resumePath}")
        return

    try:
        resumeLoader = PyPDFLoader(resumePath)
        resumeDoc = resumeLoader.load()
        for doc in resumeDoc:
            doc.metadata["source_name"] = "resume"
            doc.metadata["source_file"] = os.path.basename(resumePath)
        return resumeDoc
    except Exception as e:
        print(f"Error: Resume was not able to be loaded: {e}")
        return


def _get_public_repo_details(username: str) -> List[str]:
    api_url = f"https://api.github.com/users/{username}/repos"

    response = requests.get(api_url, params={"type": "owner", "per_page": 100 })
    response.raise_for_status()

    repos = response.json()

    non_forked_urls = [
        (repo["clone_url"], repo["default_branch"]) for repo in repos 
        if not repo["fork"] and repo["clone_url"]
    ]
    
    return non_forked_urls

def _cleanup_temp_dir(dir) -> bool:
    if not os.path.exists(dir):
        print(f"Error: cannot cleanup directory that doesn't exist: {dir}")
        return False
    
    try:
        shutil.rmtree(dir)
        print(f"Sucess: Cleaned up path: {dir}")
        return True
    except Exception as e:
        print(f"Error: Error cleaning up temporary directory {dir}: {e}")
        return False
    

def load_github_readmes(username: str, clone_dir: str) -> List[Document]:
    repo_details = _get_public_repo_details(username)

    readme_docs = []

    if not repo_details:
        print(f"No Repositories found for f{username}")
        return []
    
    for (url, branch) in repo_details:
        repo_name = url.split('/')[-1]
        local_repo_path = os.path.join(clone_dir, repo_name)

        try:
            print(f"Attempting to load README from: {url} (branch: {branch})")
            # This lambda function is the filter
            readme_loader = GitLoader(
                repo_path=local_repo_path,
                clone_url=url,
                file_filter=lambda file_path: file_path.lower().endswith("readme.md"),
                branch = branch
            )

            repo_docs = readme_loader.load()

            if repo_docs == []:
                print(f"No README found for {url} in branch {branch}")

            for doc in repo_docs:
                doc.metadata["source_type"] = "github"
                doc.metadata["repo_name"] = repo_name
                doc.metadata["source_url"] = url

            readme_docs.extend(repo_docs)
        except Exception as e:
            print(f"Could not load repo: {repo_name}: {e}")
    
    _cleanup_temp_dir(clone_dir)

    print(f"Loaded {len(readme_docs)} README.md files from GitHub.")
    return readme_docs