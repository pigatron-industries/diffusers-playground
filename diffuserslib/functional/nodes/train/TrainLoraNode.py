from matplotlib.pyplot import step
from diffuserslib.functional.FunctionalNode import FunctionalNode, WorkflowProgress
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.nodes.diffusers.ImageDiffusionNode import ModelsFuncType, ModelsType
from diffuserslib.inference import DiffusersPipelines
from diffuserslib.util import CommandProcess
import os
import shutil
import glob
import re
import yaml

ListStringFuncType = List[str] | Callable[[], List[str]]


class TrainLoraNode(FunctionalNode):
    def __init__(self, 
                 model:ModelsFuncType,
                 loraname:StringFuncType,
                 keyword:StringFuncType,
                 classword:StringFuncType,
                 train_files:StringFuncType,
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
        self.addParam("model", model, ModelsType)
        self.addParam("loraname", loraname, str)
        self.addParam("keyword", keyword, str)
        self.addParam("classword", classword, str)
        self.addParam("train_files", train_files, List[str])
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


    def process(self, model:ModelsType, loraname:str, keyword:str, classword:str, train_files:List[str]|str, output_dir:str, repeats:int, resolution:int, 
                batch_size:int, gradient_accumulation_steps:int, save_steps:int, train_steps:int, learning_rate:float, learning_rate_schedule:str, 
                learning_rate_warmup_steps, seed):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines not initialized")

        temp_train_dir = "./train"
        os.makedirs(temp_train_dir, exist_ok=True)
        shutil.rmtree(temp_train_dir)

        base_model = DiffusersPipelines.pipelines.getModel(model[0].name).base
        output_dir = os.path.join(output_dir, base_model, loraname)
        os.makedirs(output_dir, exist_ok=True)

        temp_data_dir = os.path.join(temp_train_dir, "data")
        temp_resume_dir = os.path.join(temp_train_dir, "resume")
        temp_output_dir = os.path.join(temp_train_dir, "output")

        if(isinstance(train_files, str)):
            train_files = [train_files]
        self.copyTrainingData(temp_data_dir, keyword, classword, repeats, train_files)
        resume_steps, temp_resume_dir = self.copyResumeData(temp_resume_dir, output_dir)

        command = ["accelerate", "launch", "./workspace/sd-scripts/sdxl_train_network.py"]
        command.append(f'--network_module="networks.lora"')
        command.append(f'--pretrained_model_name_or_path={model[0].name}')
        command.append(f"--train_data_dir={temp_data_dir}")
        command.append(f"--output_dir={temp_output_dir}")
        command.append(f"--resolution={resolution}")
        # command.append(f"--enable_bucket")
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
        command.append(f"--save_last_n_steps_state=0")
        if(resume_steps is not None):     
            print (f"Resuming from {temp_resume_dir}, steps={resume_steps}")
            command.append(f"--resume={temp_resume_dir}")
        
        process = CommandProcess(command)
        process.runSync()

        print("Copying output files to output dir...")
        self.copyOutputFiles(temp_output_dir, output_dir, resume_steps, loraname, keyword, classword)
        self.saveParameters(output_dir=output_dir, model=model[0].name, keyword=keyword, classword=classword, train_files=train_files, 
                            repeats=repeats, resolution=resolution, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps, 
                            save_steps=save_steps, train_steps=train_steps, learning_rate=learning_rate, learning_rate_schedule=learning_rate_schedule, 
                            learning_rate_warmup_steps=learning_rate_warmup_steps, seed=seed)
        self.copyResumeDataToOutput(resume_steps, temp_output_dir, output_dir)
        print("Done")


    
    def copyTrainingData(self, temp_data_dir:str, keyword:str, classword:str, repeats:int, train_files:List[str]):
        temp_data_dir = os.path.join(temp_data_dir, f"{repeats}_{keyword} {classword}")
        os.makedirs(temp_data_dir, exist_ok=True)
        for file_pattern in train_files:
            # TODO change * to *.png etc
            files = glob.glob(file_pattern)
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, temp_data_dir)
        
    
    def copyResumeData(self, temp_resume_dir:str, output_dir:str) -> tuple[int|None, str|None]:
        os.makedirs(temp_resume_dir, exist_ok=True)
        resume_dir_name, steps = self.findLatestSavedStateFolder(output_dir)
        if resume_dir_name is not None and steps is not None:
            temp_resume_dir = os.path.join(temp_resume_dir, resume_dir_name)
            resume_dir = os.path.join(output_dir, resume_dir_name)
            shutil.copytree(resume_dir, temp_resume_dir)
            return steps, temp_resume_dir
        return None, None
    

    def saveParameters(self, output_dir, **kwargs):
        params_file = os.path.join(output_dir, "parameters.yml")
        with open(params_file, "w") as f:
            yaml.dump(kwargs, f)


    def copyResumeDataToOutput(self, resume_steps:int|None, temp_output_dir:str, output_dir:str):
        state_dir, steps = self.findLatestSavedStateFolder(temp_output_dir)
        if steps is not None and state_dir is not None:
            total_steps = resume_steps + steps if resume_steps is not None else steps
            new_state_dir = os.path.join(output_dir, f"at-step{total_steps}-state")
            shutil.copytree(os.path.join(temp_output_dir, state_dir), new_state_dir)


    def findLatestSavedStateFolder(self, output_dir:str) -> tuple[str|None, int|None]:
        """folder name in format at-step00001000-state"""
        saved_state_folders = glob.glob(os.path.join(output_dir, "*-state"))
        latest_dir = None
        highest_steps = -1
        for dir in saved_state_folders:
            dir_name = os.path.basename(dir)
            try:
                steps = self.getStepsFromName(dir_name)
                if (steps is not None and steps > highest_steps):
                    highest_steps = steps
                    latest_dir = dir_name
            except:
                continue
        if highest_steps == -1:
            return None, None
        return latest_dir, highest_steps
    

    def copyOutputFiles(self, temp_output_dir:str, output_dir:str, resume_steps:int|None, name:str, keyword:str, classword:str):
        os.makedirs(output_dir, exist_ok=True)
        output_files = glob.glob(os.path.join(temp_output_dir, "*.safetensors"))
        for file in output_files:
            steps = self.getStepsFromName(file)
            if(steps is not None):
                total_steps = resume_steps + steps if resume_steps is not None else steps
                new_file_name = os.path.join(temp_output_dir, f"{name}_{keyword}_{classword}_{total_steps}.safetensors")
                os.rename(file, new_file_name)
                shutil.copy(new_file_name, output_dir)


    def getStepsFromName(self, name:str) -> int|None:
        """Extract number of steps from name in format at-step00001000-state or at-step00001000-state"""
        match = re.search(r"[^0-9](\d+)[^0-9]", name)
        if match:
            return int(match.group(1))
        else:
            return None


    def getProgress(self) -> WorkflowProgress|None:
        # TODO
        return WorkflowProgress(0, None)