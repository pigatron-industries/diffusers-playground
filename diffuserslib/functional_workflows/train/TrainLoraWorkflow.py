from cv2 import repeat
from diffuserslib.functional import *
from diffuserslib.functional.nodes.train.TrainLoraNode import TrainLoraNode


class TrainLoraWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Train Lora", None, workflow=True, subworkflow=False)

    def build(self):
        model_input = DiffusionModelUserInputNode()
        keyword_input = StringUserInputNode(value = "", name="keyword")
        classword_input = StringUserInputNode(value = "", name="classword")
        repeats_input = IntUserInputNode(value = 10, name="repeats")
        resolution_input = IntUserInputNode(value = 768, name="resolution")
        save_steps_input = IntUserInputNode(value = 100, name="save_steps")
        train_steps_input = IntUserInputNode(value = 1000, name="train_steps")
        seed_input = SeedUserInputNode(value = None, name="seed")

        train_files_input = StringUserInputNode(value = "", name="train_files")
        output_dir_input = StringUserInputNode(value = "", name="output_dir")
        
        
        train_lora = TrainLoraNode(model = model_input,
                                   keyword = keyword_input,
                                   classword = classword_input,
                                   train_files = train_files_input,
                                   output_dir = output_dir_input,
                                   repeats = repeats_input,
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
    
