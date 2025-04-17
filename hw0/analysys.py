import os
import re

def read_file(filename):
    """Reads the contents of a file into a list."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:  # Specify encoding for broader compatibility
            wordlist = []
            for line in f:
                words = line.strip().split()
                wordlist.extend(words)
        return wordlist
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def sort_word_list(word_list):
    """Sorts a list of words alphabetically."""
    word_list.sort()  # Sort in place (modifies the original list)
    return word_list


def count_word_forms(word_list):
    """Counts the occurrences of each word form in a list."""
    count = {}
    for word in word_list:
        first_letter = word[0].lower()  # Case-insensitive
        if first_letter in count:
            count[first_letter] += 1
        else:
            count[first_letter] = 1
    return count


def find_frequent_suffixes(suffix_to_forms):
    """Finds frequent suffixes and their forms."""
    print("Suffix to Forms Dictionary:")
    for suffix, forms in suffix_to_forms.items():
        print(f"{suffix}: {forms}")


def process_file(filename):
    """Processes the file, reads it, extracts word forms, sorts, counts, and finds frequent suffixes."""


    wordlist = read_file(filename)
    if wordlist is None:
        return


    sorted_word_list = sort_word_list(wordlist)
    print("Sorted Word List:")
    for word in sorted_word_list:
        print(word)  # Print each word


    suffix_to_forms = {}
    for word in sorted_word_list:
      if word[0].lower() not in suffix_to_forms:
          suffix_to_forms[word[0].lower()] = set() #Store as a set to avoid duplicates.
      suffix_to_forms[word[0].lower()].add(word)  # Add to the set


    print("Suffix to Forms Dictionary:")
    for suffix, forms in suffix_to_forms.items():
        print(f"{suffix}: {forms}")


def save_results(suffix_candidates, output_filename):
    """Saves the results to a file."""

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:  # Specify encoding for broader compatibility
            print(f"Saving results to '{output_filename}'")
            for suffix in suffix_candidates:
                f.write(suffix + '\n')

    except Exception as e:
        print(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    # Example Usage:  Create a dummy de-word-forms.txt file
    # This is just for demonstrating the script's structure, you can replace with your actual data


    dummy_file = "de-word-forms.txt"
    with open(dummy_file, 'w', encoding='utf-8') as f:
        f.write("ich\n")
        f.write("sie\n")
        f.write("das\n")
        f.write("ist\n")
        f.write("du\n")
        f.write("eins\n")
        f.write("zwei\n")
        f.write("drei\n")



    # Call the main functions:
    read_file(dummy_file)  # Read the file contents
    sort_word_list(read_file(dummy_file))
    count_word_forms(read_file(dummy_file))
    find_frequent_suffixes(read_file(dummy_file))
    save_results(read_file(dummy_file), "suffix_candidates.txt")  # Save the results to a file

