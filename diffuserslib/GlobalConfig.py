

class GlobalConfig:
    device = "mps"
    inputs_dirs = []
    embeddings_dirs = []
    loras_dirs = []
    workflow_dirs = []
    workflowstate_dirs = []
    modelconfigs = {}

    # MiniMax-H3 (Apple Silicon) - checkout of minimax-h3-mac and its python interpreter.
    # Model weights are expected under {minimax_h3_dir}/models/... and the generation
    # entrypoint at {minimax_h3_dir}/scripts/generate.py.
    minimax_h3_proj_dir = "./workspace/minimax-h3-mac"
    minimax_h3_model_dir = "/Volumes/T9/models/other/minimax-h3"


    @staticmethod   
    def getModelsByBase(type:str, base:str):
        print(GlobalConfig.modelconfigs)
        return GlobalConfig.modelconfigs[type][base]["models"]
    
    
    @staticmethod
    def addModelConfig(modelconfig:dict):
        for type in modelconfig:
            if type not in GlobalConfig.modelconfigs:
                GlobalConfig.modelconfigs[type] = {}
            try:
                for base in modelconfig[type]:
                    print(base)
                    if base not in GlobalConfig.modelconfigs[type]:
                        GlobalConfig.modelconfigs[type][base] = {}
                        GlobalConfig.modelconfigs[type][base]["models"] = []
                    GlobalConfig.modelconfigs[type][base]["models"].extend(modelconfig[type][base]["models"])
            except Exception as e:
                pass
            
            