import glob
import os
from typing import List

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

def getFileList(rootDir, patterns:List[str], recursive=False):
    patterns = [f"{rootDir}/{pattern}" for pattern in patterns]
    if recursive:
        patterns += [f"{rootDir}/**/{pattern}" for pattern in patterns]
    filelist = []
    for pattern in patterns:
        filelist += glob.glob(pattern, recursive=recursive)
    for file in filelist:
        file = os.path.relpath(file, rootDir)
    return filelist