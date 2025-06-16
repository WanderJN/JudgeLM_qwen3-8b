import json

def jlload(f, mode='r'):
    data_list = []
    # Open the JSONL file
    with open(f, mode) as file:
        # Iterate through each line in the file
        for line in file:
            # Parse the line as a JSON object
            data = json.loads(line)
            # You can now work with the JSON object (e.g., print it, extract data from it, etc.)
            data_list.append(data)

    return data_list


def extract_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    return data_list
