import os
import subprocess

original_location = os.getcwd()
script_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_directory, "../external")
full_folder_path = os.path.abspath(folder_path)
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)

libs = ["https://github.com/google/googletest.git"]

for lib in libs:
    subprocess.run(["git", "clone", lib], cwd=full_folder_path)
