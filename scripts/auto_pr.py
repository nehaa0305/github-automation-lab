from github import Github
from dotenv import load_dotenv
import os

# === LOAD TOKEN FROM .env ===
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")


REPO_NAME = "nehaa0305/github-automation-lab"  
SOURCE_BRANCH = "feature/login"
TARGET_BRANCH = "main"
PR_TITLE = "Auto: Create PR from feature branch"
PR_BODY = "This PR was automatically created from Python script."


g = Github(TOKEN)
repo = g.get_repo(REPO_NAME)


open_prs = repo.get_pulls(state="open", head=f"{repo.owner.login}:{SOURCE_BRANCH}")
if open_prs.totalCount > 0:
    print(f"PR already exists for {SOURCE_BRANCH}")
else:
  
    pr = repo.create_pull(
        title=PR_TITLE,
        body=PR_BODY,
        head=SOURCE_BRANCH,
        base=TARGET_BRANCH,
    )
    print(f"Created PR: {pr.html_url}")
