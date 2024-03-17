from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.util import CommandProcess


class TrainLoraNode(FunctionalNode):
    def __init__(self, 
                 model:StringFuncType,
                 keyword:StringFuncType,
                 classword:StringFuncType,
                 train_data_dir:StringFuncType,
                 output_dir:StringFuncType,
                 repeats:IntFuncType = 10,
                 resolution:IntFuncType = 768,
                 batch_size:IntFuncType = 1,
                 gradient_accumulation_steps:IntFuncType = 1,
                 save_steps:IntFuncType = 100,
                 train_steps:IntFuncType = 1000,
                 learning_rate:FloatFuncType = 0.0001,
                 learning_rate_schedule:StringFuncType = "constant",
                 learning_rate_warmup_steps:IntFuncType = 0,
                 seed:IntFuncType = 0,
                 name:str = "train_lora",
                 ):
        super().__init__(name)
        self.addParam("model", model, str)
        self.addParam("keyword", keyword, str)
        self.addParam("classword", classword, str)
        self.addParam("train_data_dir", train_data_dir, str)
        self.addParam("output_dir", output_dir, str)
        self.addParam("repeats", repeats, int)
        self.addParam("resolution", resolution, int)
        self.addParam("batch_size", batch_size, int)
        self.addParam("gradient_accumulation_steps", gradient_accumulation_steps, int)
        self.addParam("save_steps", save_steps, int)
        self.addParam("train_steps", train_steps, int)
        self.addParam("learning_rate", learning_rate, float)
        self.addParam("learning_rate_schedule", learning_rate_schedule, str)
        self.addParam("learning_rate_warmup_steps", learning_rate_warmup_steps, int)
        self.addParam("seed", seed, int)


    def process(self, model, keyword, classword, train_data_dir, output_dir, repeats, resolution, batch_size, gradient_accumulation_steps, 
                save_steps, train_steps, learning_rate, learning_rate_schedule, learning_rate_warmup_steps, seed):
        
        
        # TODO: copy input data to temp dir in format "repeats_keyword classword"
        # TODO: resize input data?
        # TODO: create temp output dir
        # TODO: check output dir for existing saved state and resume training, look for sub folder like at-step00001000-state
        # TODO: record starting step based on existing saved state folder name
        # TODO: rename output files to something sensible including keyword and classword and steps
        # TODO: save input parameters to output dir


        command = ["accelerate", "launch", "sdxl_train_network.py"]
        command.append(f'--network_module="networks.lora"')
        command.append(f'--pretrained_model_name_or_path={model}')
        command.append(f"--train_data_dir={train_data_dir}")
        command.append(f"--output_dir={output_dir}")
        command.append(f"--resolution={resolution}")
        command.append(f"--train_batch_size={batch_size}")
        command.append(f"--gradient_accumulation_steps={gradient_accumulation_steps}")
        command.append(f"--save_every_n_steps={save_steps}")
        command.append(f"--max_train_steps={train_steps}")
        command.append(f"--learning_rate={learning_rate}")
        command.append(f"--lr_scheduler={learning_rate_schedule}")
        command.append(f"--lr_warmup_steps={learning_rate_warmup_steps}")
        command.append(f"--seed={seed}")
        command.append(f"--lowram")
        command.append(f"--save_state")
        
        process = CommandProcess(command)
        process.runSync()

    