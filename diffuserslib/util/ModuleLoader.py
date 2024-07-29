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
            if(os.path.basename(file) == '__init__.py'):
                modules.append(ModuleLoader.load_from_file(file, parent_module = path))
        for file in filelist:
            if(os.path.basename(file) != '__init__.py'):
                modules.append(ModuleLoader.load_from_file(file, parent_module = path))
        return modules
            

    @staticmethod
    def load_from_file(path, parent_module = None, module_name = None):
        if(module_name is None):
            module_name = os.path.splitext(os.path.basename(path))[0]
            if(parent_module is not None):
                parent_dir = os.path.dirname(parent_module)
                relative_path = os.path.relpath(path, parent_dir)
                directories = os.path.dirname(relative_path).split(os.sep)
                module_name = '.'.join(directories + [module_name])
                if(module_name.endswith('.__init__')):
                    module_name = module_name[:-9]
        print(module_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if (spec is not None and spec.loader is not None):              
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
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