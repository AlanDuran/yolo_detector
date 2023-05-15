import os
import string


# Removes invalid characters from a file name
def normalize_filenames(path):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    # convert into an immutable set for performance reasons
    valid_chars = frozenset(valid_chars)

    # iterate over files in that directory
    for root, dirs, files in os.walk(path):

        for file in files:
            fixed_name = ''.join(c for c in file if c in valid_chars)

            if fixed_name != file:
                initial_path = os.path.join(root, file)
                fixed_path = os.path.join(root, fixed_name)
                print("Found invalid filename %s. Replacing with %s"
                      % (initial_path, fixed_path))
                os.rename(initial_path, fixed_path)


def count_classes(path, skip_repeats=True):
    class_counts = {}

    # Iterate over each file in the directory
    for root, dirs, files in os.walk(path):

        for filename in files:
            if not filename.endswith('.txt'):
                continue

            # Read the contents of the file
            with open(os.path.join(root, filename), 'r') as f:
                file_contents = f.read()

            class_in_file = {}

            # Split the file contents by newline characters and iterate over each line
            for line in file_contents.split('\n'):
                # Skip empty lines
                if not line:
                    continue

                # Split the line by whitespace characters and extract the class label
                class_label = line.split()[0]

                if class_label in class_in_file:
                    class_in_file[class_label] += 1
                else:
                    class_in_file[class_label] = 1

                # Increment the count for the class label in the dictionary
                if class_in_file[class_label] == 1 or not skip_repeats:
                    if class_label in class_counts:
                        class_counts[class_label] += 1
                    else:
                        class_counts[class_label] = 1

    print(class_counts)

