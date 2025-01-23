from github import Github, GithubException
import os
import base64

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"  # Replace with your username
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "finalofc"  # Replace with your repository name
AI_PROGRESS_FILENAME = "cfr_data.pkl"

def save_progress_to_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("AI_PROGRESS_TOKEN not set. Progress saving disabled.")
        return

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            contents = repo.get_contents(filename, ref="main")  # Specify the main branch
            with open(filename, 'rb') as f:
                content = f.read()
            repo.update_file(contents.path, "Update AI progress", base64.b64encode(content).decode('utf-8'), contents.sha, branch="main") # Encode content to base64
            print(f"AI progress saved to GitHub: {GITHUB_REPOSITORY}/{filename}")
        except GithubException as e:
            if e.status == 404:
                with open(filename, 'rb') as f:
                    content = f.read()
                repo.create_file(filename, "Initial AI progress", base64.b64encode(content).decode('utf-8'), branch="main") # Encode content to base64
                print(f"Created new file for AI progress on GitHub: {GITHUB_REPOSITORY}/{filename}")
            else:
                print(f"Error saving progress to GitHub (other than 404): {e}")
    except GithubException as e:
        print(f"Error saving progress to GitHub: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


def load_progress_from_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("AI_PROGRESS_TOKEN not set. Progress loading disabled.")
        return

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        contents = repo.get_contents(filename, ref="main") # Specify the main branch
        file_content = base64.b64decode(contents.content) # Decode from base64
        with open(filename, 'wb') as f:
            f.write(file_content)
        print(f"AI progress loaded from GitHub: {GITHUB_REPOSITORY}/{filename}")

    except GithubException as e:
        if e.status == 404:
            print("Progress file not found in GitHub repository.")
        else:
            print(f"Error loading progress from GitHub: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
