import os
import zipfile
import xml.etree.ElementTree as ET
import re
import multiprocessing
from tqdm import tqdm
import requests

def process_law(law, output_directory):
    """
    Function to process each item from the item array. It does the following for each item.
    """
    # Download the zip file
    item_response = requests.get(law['link'], stream=True, timeout=60)
    zip_name = re.sub(r'\W+', '', law['link']) + '.zip'
    zip_path = os.path.join(output_directory, zip_name)
    with open(zip_path, 'wb') as file_driver:
        for chunk in item_response.iter_content(chunk_size=128):
            file_driver.write(chunk)
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.xml'):
                zip_ref.extract(file_name, output_directory)
    # Remove the zip file
    os.remove(zip_path)
    return 1

def process_law_wrapper(args):
    """
    Wrapper function to use a non-lambda function with multiprocessing.
    """
    return process_law(*args)

def main(output_directory='./de_federal_raw'):
    """
    Download the XML file with all laws and run process_law for each one.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Download the XML file
    response = requests.get('https://www.gesetze-im-internet.de/gii-toc.xml', timeout=10)

    # Parse the XML from the response text
    root = ET.fromstring(response.content)

    # Create an array of dictionaries
    item_array = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        link = item.find('link').text
        item_dict = {'title': title, 'link': link}
        item_array.append(item_dict)

    # Set the number of items to process
    num_items_to_process = len(item_array)  # change this to control how many items to process
    print(f"Processing {num_items_to_process} items out of {len(item_array)} total items")

    # Initialize a Pool with the number of available processors
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    print(f"Using {multiprocessing.cpu_count()} cores/processes in parallel")

    # Use Pool's map function to process the items in parallel
    with tqdm(total=num_items_to_process, desc="Processing files", dynamic_ncols=True) as pbar:
        for _ in pool.imap_unordered(process_law_wrapper, [(law, output_directory) for law in item_array[:num_items_to_process]]):
            pbar.update()

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Specify your desired output directory here
    custom_output_directory = '/Users/therealchrisbrennan/Documents/Project/JurAI/Gesetze/de_federal_law'

    # Run the script with the specified output directory
    main(output_directory=custom_output_directory)
