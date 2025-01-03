from diffuserslib import GlobalConfig
from diffuserslib.functional import *
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.train.TrainLoraNode import TrainLoraNode
from diffuserslib.functional.nodes.image.diffusers.user.DiffusionModelUserInputNode import DiffusionModelUserInputNode
from diffuserslib.functional.nodes.train import *


class TrainLoraWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Train Lora", None, workflow=True, subworkflow=False)

    def build(self):
        script_input = ListSelectUserInputNode(value = "", options = TrainLoraNode.TRAIN_SCRIPTS, name="script")
        model_input = DiffusionModelUserInputNode()
        loraname_input = StringUserInputNode(value = "", name="loraname")
        keyword_input = StringUserInputNode(value = "", name="keyword")
        classword_input = StringUserInputNode(value = "", name="classword")
        resolution_input = IntUserInputNode(value = 768, name="resolution")
        bucket_input = BoolUserInputNode(value = False, name="enable_bucket")
        network_dim_input = IntUserInputNode(value = 4, name="network_dim")
        network_alpha_input = IntUserInputNode(value = 1, name="network_alpha")
        save_steps_input = IntUserInputNode(value = 100, name="save_steps")
        train_steps_input = IntUserInputNode(value = 1000, name="train_steps")
        learning_rate_input = FloatUserInputNode(value = 0.0001, format='%.5f', name="learning_rate")
        seed_input = SeedUserInputNode(value = None, name="seed")

        train_data_input = TrainDataUserInputNode(name="train_data", repeats=True)

        output_dir_input = ListSelectUserInputNode(value = "", options=GlobalConfig.loras_dirs, name="output_dir")
        
        train_lora = TrainLoraNode(trainscript = script_input, 
                                   model = model_input,
                                   loraname = loraname_input,
                                   keyword = keyword_input,
                                   classword = classword_input,
                                   train_data = train_data_input,
                                   output_dir = output_dir_input,
                                   resolution = resolution_input,
                                   enable_bucket = bucket_input,
                                   batch_size = 1,
                                   gradient_accumulation_steps = 1,
                                   save_steps = save_steps_input,
                                   train_steps = train_steps_input,
                                   learning_rate = learning_rate_input,
                                   learning_rate_schedule = "constant",
                                   learning_rate_warmup_steps = 0,
                                   seed = seed_input,
                                   network_dim = network_dim_input,   
                                   network_alpha = network_alpha_input,
                                   name = "train_lora")
        
        return train_lora
    
