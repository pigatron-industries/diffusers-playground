from diffuserslib import GlobalConfig
from diffuserslib.functional import *
from diffuserslib.functional.nodes.train.TrainLoraNode import TrainLoraNode


class TrainLoraWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Train Lora", None, workflow=True, subworkflow=False)

    def build(self):
        model_input = DiffusionModelUserInputNode()
        loraname_input = StringUserInputNode(value = "", name="loraname")
        keyword_input = StringUserInputNode(value = "", name="keyword")
        classword_input = StringUserInputNode(value = "", name="classword")
        resolution_input = IntUserInputNode(value = 768, name="resolution")
        save_steps_input = IntUserInputNode(value = 100, name="save_steps")
        train_steps_input = IntUserInputNode(value = 1000, name="train_steps")
        seed_input = SeedUserInputNode(value = None, name="seed")

        train_data_input = TrainDataUserInputNode(name="train_data")

        output_dir_input = ListSelectUserInputNode(value = "", options=GlobalConfig.loras_dirs, name="output_dir")
        
        train_lora = TrainLoraNode(model = model_input,
                                   loraname = loraname_input,
                                   keyword = keyword_input,
                                   classword = classword_input,
                                   train_data = train_data_input,
                                   output_dir = output_dir_input,
                                   resolution = resolution_input,
                                   batch_size = 1,
                                   gradient_accumulation_steps = 1,
                                   save_steps = save_steps_input,
                                   train_steps = train_steps_input,
                                   learning_rate = 0.0001,
                                   learning_rate_schedule = "constant",
                                   learning_rate_warmup_steps = 0,
                                   seed = seed_input,
                                   name = "train_lora")
        
        return train_lora
    
