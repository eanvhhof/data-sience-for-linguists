
file_path = "sq-sample.txt"
def get_unique_chars(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    unique_chars = {char.lower() for char in text if char.isalnum()}  # Extract letters and numbers
    return sorted(unique_chars)  # Return sorted list of unique characters

  # Replace with the actual file path
unique_chars = get_unique_chars(file_path)
print(unique_chars)
