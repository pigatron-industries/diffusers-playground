import glob
import os

def getPathsFiles(pattern):
    pathsfiles = []
    for path in sorted(glob.glob(pattern)):
        file = os.path.basename(os.path.normpath(path))
        pathsfiles.append((path, file))
    return pathsfiles