import glob
import os

def getPathsFiles(pattern):
    pathsfiles = []
    for path in sorted(glob.glob(pattern)):
        file = os.path.basename(os.path.normpath(path))
        pathsfiles.append((path, file))
    return pathsfiles

def getLeafFolders(root_folder):
    leaf_folders = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        if not subfolders:
            leaf_folders.append(foldername)
    return leaf_folders