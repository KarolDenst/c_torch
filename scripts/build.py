import os
import subprocess
import argparse

original_location = os.getcwd()
script_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_directory, "../build")
full_folder_path = os.path.abspath(folder_path)
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)

parser = argparse.ArgumentParser(description="A script to build the project.")

parser.add_argument("--debug", help="Build in release mode.", action="store_true")
parser.add_argument(
    "--run", help="Run the program after the build.", action="store_true"
)
parser.add_argument(
    "--test", help="Run the tests after the build.", action="store_true"
)

args = parser.parse_args()

if not args.debug:
    subprocess.run(
        ["cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", ".."],
        check=True,
        cwd=full_folder_path,
    )
else:
    subprocess.run(
        ["cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Debug", ".."],
        check=True,
        cwd=full_folder_path,
    )

subprocess.run(["ninja"], check=True, cwd=full_folder_path)

if args.run:
    subprocess.run([".\\TensorC.exe"], check=True, cwd=full_folder_path, shell=True)

if args.test:
    subprocess.run(
        [".\\tests\\tests.exe"], check=True, cwd=full_folder_path, shell=True
    )
