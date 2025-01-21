from github import Github
import os
import base64

def save_progress_to_github(filename):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("Переменная AI_PROGRESS_TOKEN не установлена. Сохранение прогресса невозможно.")
        return

    g = Github(token)
    repo = g.get_user("Azerus96").get_repo("finalofc")

    try:
        contents = repo.get_contents(filename)
        with open(filename, 'rb') as f:
            content = f.read()
        repo.update_file(contents.path, "Update AI progress", content, contents.sha)
        print("Прогресс ИИ успешно сохранен на GitHub.")
    except Exception as e:
        if e.status == 404:
            with open(filename, 'rb') as f:
                content = f.read()
            repo.create_file(filename, "Initial AI progress", content)
            print("Создан новый файл для сохранения прогресса ИИ на GitHub.")
        else:
            print(f"Ошибка при сохранении прогресса на GitHub: {e}")

def load_progress_from_github(filename):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("Переменная AI_PROGRESS_TOKEN не установлена. Загрузка прогресса невозможна.")
        return

    g = Github(token)
    repo = g.get_user("Azerus96").get_repo("finalofc")

    try:
        contents = repo.get_contents(filename)
        file_content = base64.b64decode(contents.content)
        with open(filename, 'wb') as f:
            f.write(file_content)
        print("Прогресс ИИ успешно загружен с GitHub.")
    except Exception as e:
        print(f"Ошибка при загрузке прогресса с GitHub: {e}")
