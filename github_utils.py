from github import Github
import os
import base64

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "finalofc"
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
            contents = repo.get_contents(filename)
            with open(filename, 'rb') as f:
                content = f.read()
            repo.update_file(contents.path, "Update AI progress", content, contents.sha)
            print(f"AI progress saved to GitHub: {GITHUB_REPOSITORY}/{filename}")
        except GithubException as e:
            if e.status == 404:
                with open(filename, 'rb') as f:
                    content = f.read()
                repo.create_file(filename, "Initial AI progress", content)
                print(f"Created new file for AI progress on GitHub: {GITHUB_REPOSITORY}/{filename}")
            else:
                raise  # Re-raise the exception if it's not a 404

    except GithubException as e:
        print(f"Error saving progress to GitHub: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_progress_from_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("AI_PROGRESS_TOKEN not set. Progress loading disabled.")
        return

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        contents = repo.get_contents(filename)
        file_content = base64.b64decode(contents.content)
        with open(filename, 'wb') as f:
            f.write(file_content)
        print(f"AI progress loaded from GitHub: {GITHUB_REPOSITORY}/{filename}")

    except GithubException as e:
        print(f"Error loading progress from GitHub: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
