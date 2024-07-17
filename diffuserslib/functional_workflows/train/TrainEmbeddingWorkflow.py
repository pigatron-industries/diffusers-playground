from diffuserslib import GlobalConfig
from diffuserslib.functional import *
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.train.TrainEmbeddingNode import TrainEmbeddingNode
from diffuserslib.functional.nodes.image.diffusers.user.DiffusionModelUserInputNode import DiffusionModelUserInputNode
from diffuserslib.functional.nodes.train import *


class TrainEmbeddingWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Train Embedding", None, workflow=True, subworkflow=False)

    def build(self):
        model_input = DiffusionModelUserInputNode()
        embeddingname_input = StringUserInputNode(value = "", name="embeddingname")
        keyword_input = StringUserInputNode(value = "", name="keyword")
        classword_input = StringUserInputNode(value = "", name="classword")
        initword_input = StringUserInputNode(value = "", name="initword")
        num_vectors_per_token = IntUserInputNode(value = 1, name="num_vectors_per_token")
        template_type_input = ListSelectUserInputNode(value = "object", options=["object", "style"], name="template_type")
        resolution_input = IntUserInputNode(value = 768, name="resolution")
        bucket_input = BoolUserInputNode(value = False, name="enable_bucket")
        save_steps_input = IntUserInputNode(value = 100, name="save_steps")
        train_steps_input = IntUserInputNode(value = 1000, name="train_steps")
        learning_rate_input = FloatUserInputNode(value = 5.0e-04, format='%.5f', name="learning_rate")
        seed_input = SeedUserInputNode(value = None, name="seed")
        save_state_input = BoolUserInputNode(name="save_state")

        train_data_input = TrainDataUserInputNode(name="train_data", repeats=True)

        output_dir_input = ListSelectUserInputNode(value = "", options=GlobalConfig.embeddings_dirs, name="output_dir")
        
        train_lora = TrainEmbeddingNode(model = model_input,
                                   embeddingname = embeddingname_input,
                                   keyword = keyword_input,
                                   classword = classword_input,
                                   initword = initword_input,
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
                                   num_vectors_per_token=num_vectors_per_token,
                                   template_type=template_type_input,
                                   save_state = save_state_input,
                                   name = "train_embedding")
        
        return train_lora
    
