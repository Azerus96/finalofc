import pickle
import os

def save_data(data, filename):
    """Saves data to a file using pickle."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")
    except (pickle.PickleError, OSError, Exception) as e:  # Catching a wider range of exceptions
        print(f"Error saving data to {filename}: {e}")


def load_data(filename):
    """Loads data from a file using pickle."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(f"Data loaded from {filename}")
            return data
    except (pickle.PickleError, OSError, EOFError, Exception) as e:  # Catching a wider range of exceptions
        print(f"Error loading data from {filename}: {e}")
        return None
