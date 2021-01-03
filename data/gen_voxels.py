from zipfile import ZipFile
import os
import sys
import urllib.request
import shutil

# Platform warning
if sys.platform != "win32":
    print("You need to modify some of the code in this file: see the README for more info.")
    quit()

# check for and download files

# Windows specific
if not os.path.exists("binvox.exe"):
    print("Downloading object voxelizer...")
    binvox = "https://www.patrickmin.com/binvox/win/binvox.exe?rnd=1609553150930888"
    urllib.request.urlretrieve(binvox, "binvox.exe")

if not os.path.exists("03001627.zip"):
    chairs = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/03001627.zip"
    answer = input("Chairs dataset (03001627.zip) is missing! Large download (1.4G): Continue? (Y/N)")
    if answer.strip()[0].lower() == 'y':
        print("Downloading... this might take a while.")
        urllib.request.urlretrieve(chairs, "03001627.zip")
        print("Finished downloading.")
    else:
        print(f"Please download the ShapeNet chairs dataset at {chairs}.")
        quit()

# make chair model directory
if not os.path.isdir('chair_models/'):
    os.mkdir('chair_models/')

n_models = 1000  # number of models to extract

chairs = "03001627.zip"
with ZipFile(chairs, 'r') as zf:
    files = zf.namelist()
    objs = [file for file in files if ".obj" in file]
    for i in range(n_models):
        model_name = objs[i][9:-10]
        zf.extract(objs[i])
        os.system(f"binvox.exe -d 64 -cb {objs[i]}")  # Change this to match the name of your binvox executable
        shutil.copyfile(f"{objs[i][:-4]}.binvox", f"chair_models/{model_name}.binvox")
        shutil.rmtree(objs[i][:-9])
