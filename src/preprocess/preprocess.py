import os
import glob
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Preprocess:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n \n","\n", ".", "?", "!"],
            chunk_size=2048, 
            chunk_overlap=0,
            length_function=len,
        )
    
    def find_text(self): 
        # Use glob to recursively find all files with the specified extension
        text_files_paths = glob.glob(os.path.join(self.dir_path, '**', '*.txt'), recursive=True)
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
        with open('data/train/train_zivilB_2048.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['text'])
            for text in all_texts:
                writer.writerow([text])
    
    def find_subfolders_and_merge_files(self):
        # find all subfolders
        subfolders = [f.path for f in os.scandir(self.dir_path) if f.is_dir()]
        # merge all files in subfolders
        for subfolder in subfolders:
            print(subfolder)
            os.chdir(subfolder)
            os.system("cat *.txt > merged.txt")
            os.chdir("/Users/kiliansprenkamp/Desktop/code/jurai")
            #  os.system("rm *.txt")
            # os.chdir(self.dir_path)
            # os.system("mv " + subfolder + "/merged.txt " + subfolder + ".txt")

                
if __name__ == "__main__":
    dir_path = "data/currentChris/GesetzeZivil_B/"
    pwd = os.getcwd()
    print(pwd)
    preprocess = Preprocess(dir_path)
    preprocess.run()
    # preprocess.find_subfolders_and_merge_files()        
        


