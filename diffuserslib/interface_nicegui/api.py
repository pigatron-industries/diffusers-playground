from nicegui import ui, app
import importlib
import importlib.util
import sys
import os
import glob


def loadWorkflows():
    path = os.path.join(os.path.dirname(__file__), '../functional_workflows')
    files = glob.glob(path + '/*.py')
    workflows = {}
    
    for file in files:
        spec = importlib.util.spec_from_file_location("module.workflow", file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module.workflow"] = module
        spec.loader.exec_module(module)
        workflows[module.name] = module
        print(f"Loaded workflow: {module.name}")

    return workflows


@app.get('/api/test')
def test():
    return {'test': 0}