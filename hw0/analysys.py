# datsci_ex00.py

# === Task 1: Reading the File Contents into a List ===
word_list = []

with open("de-word-forms.txt", "r", encoding="utf-8") as file:
    for line in file:
        word = line.strip()
        word_list.append(word)

# Print first five words to check
print("First five words:", word_list[:5])

# === Task 2: Sorting and Counting ===

# a) Alphabetical sorting
alphabetical_list = sorted(word_list)
print("\nFirst 10 words alphabetically:", alphabetical_list[:10])
print("Last 10 words alphabetically:", alphabetical_list[-10:])

# b) Count how many word forms start with which letter
print("\nWord counts by starting letter:")
current_letter = ''
count = 0

for word in alphabetical_list:
    if word[0] != current_letter:
        if current_letter != '':
            print(f"{current_letter}: {count}")
        current_letter = word[0]
        count = 1
    else:
        count += 1
# Print the last one
print(f"{current_letter}: {count}")

# === Task 3: Finding Frequent Suffixes ===

suffix_to_forms = {}

for word in word_list:
    if len(word) >= 3:
        suffix = word[-3:]
        if suffix not in suffix_to_forms:
            suffix_to_forms[suffix] = set()
        suffix_to_forms[suffix].add(word)

# Collect suffix candidates (occurring in more than 200 words)
suffix_candidates_output = []
print("\nSuffixes that occur in more than 200 word forms:")

for suffix in sorted(suffix_to_forms):
    forms = suffix_to_forms[suffix]
    if len(forms) > 200:
        sorted_forms = sorted(forms)
        line = f"{suffix}: {', '.join(sorted_forms)}"
        print(line)
        suffix_candidates_output.append(line)

# === Task 4: Saving Results to File ===

with open("suffix_candidates.txt", "w", encoding="utf-8") as out_file:
    for line in suffix_candidates_output:
        out_file.write(line + "\n")
