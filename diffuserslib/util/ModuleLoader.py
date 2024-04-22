import sys
import os
import glob
import importlib.util
import types
from dataclasses import dataclass
from diffuserslib.functional import *



class ModuleLoader:

    @staticmethod
    def load_from_directory(path, recursive = True):
        filelist = []
        if(recursive):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        filelist.append(os.path.join(root, file))
        else:
            filelist = glob.glob(path + '/*.py')
        modules = []
        for file in filelist:
            modules.append(ModuleLoader.load_from_file(file))
        return modules
            

    @staticmethod
    def load_from_file(path):
        spec = importlib.util.spec_from_file_location("module", path)
        if (spec is not None and spec.loader is not None):              
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    
    @staticmethod
    def get_vars(module):
        external_module_names = set(dir(__import__('builtins')))
        external_module_names.add('__name__')
        for name in dir(sys.modules[__name__]):
            if name != '__name__':
                external_module_names.add(name)
        module_names = dir(module)
        defined_names = [name for name in module_names if name not in external_module_names]
        vars = {}
        for name in defined_names:
            vars[name] = getattr(module, name)
        return vars