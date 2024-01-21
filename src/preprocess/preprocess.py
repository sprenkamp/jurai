import os
import glob
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Preprocess:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n \n","\n", ".", "?", "!"],
            chunk_size=512, 
            chunk_overlap=0,
            length_function=len,
        )
    
    def find_text(self):

        # Use glob to recursively find all files with the specified extension
        text_files_paths = glob.glob(os.path.join(self.dir_path, '**', '*.[tc][xs][tv]'), recursive=True)
        print(len(text_files_paths))
        # Return the list of found text files
        return text_files_paths

    def read_text(self, text_files_paths):
        # Read the text file
        with open(text_files_paths, 'r') as file:
            text = file.read()
        return text
    
    def run(self):
        all_texts = []
        text_files_paths = self.find_text()
        for text_files_path in text_files_paths:
            text = self.read_text(text_files_path)
            texts = self.text_splitter.create_documents([text])
            for text in texts:
                all_texts.append(text.page_content)
        # Save all texts as CSV
        with open('train_zivilB_gutachten.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['text'])
            for text in all_texts:
                writer.writerow([text])
        
                
                
if __name__ == "__main__":
    dir_path = "data/currentChris/GesetzeZivil_B/"
    preprocess = Preprocess(dir_path)
    preprocess.run()
        
        


