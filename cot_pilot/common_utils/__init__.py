import os

def read_lines(file_path):
    """
    Reads lines from a file and returns them as a list of strings.
    Strips whitespace from each line.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
