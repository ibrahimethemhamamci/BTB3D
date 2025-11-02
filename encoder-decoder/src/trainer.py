# -*- coding:UTF-8 -*-
from modeling.magvit_model import VisionTokenizer
from src.data import VideoDataset, ImageDataset, DataLoader
from .data_clip_old import CTReportDataset
from .data_inference_old import CTReportDatasetinfer

from src import losses
from transformers import get_cosine_schedule_with_warmup
import torch
import os
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
import json
from torchvision.io import write_video, write_png
from einops import rearrange
from modeling.ema import ModelEmaV3 as EMA
from torchvision.models import resnet50
import math
import time
from omegaconf import OmegaConf
import torch.nn.functional as F
import gc
from collections import namedtuple
from safetensors.torch import load_model
import pathlib
import random
import torch
from .patcher_module import UnPatcher3D, Patcher3D
from datetime import timedelta


from torch.utils.data import DataLoader, Sampler


class AlternatingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indices_2d = [i for i in range(len(dataset)) if i % 2 == 0]  # Even indices (2D samples)
        self.indices_3d = [i for i in range(len(dataset)) if i % 2 == 1]  # Odd indices (3D samples)
        self.current_mode = "3D"  # Start with 2D batches
        self.switch_count = 0  # Track how many batches have been yielded in current mode
        self.switch_max = 4 * int(os.environ["SLURM_NNODES"])
    def __iter__(self):
        idx_2d, idx_3d = 0, 0
        while idx_2d < len(self.indices_2d) or idx_3d < len(self.indices_3d):
            if self.current_mode == "2D" and idx_2d < len(self.indices_2d):
                yield self.indices_2d[idx_2d : idx_2d + self.batch_size]
                idx_2d += self.batch_size
            elif self.current_mode == "3D" and idx_3d < len(self.indices_3d):
                yield self.indices_3d[idx_3d : idx_3d + self.batch_size]
                idx_3d += self.batch_size

            self.switch_count += 1
            if self.switch_count >= self.switch_max:  # Switch every 4 batches
                self.current_mode = "3D" if self.current_mode == "2D" else "3D"
                self.switch_count = 0

    def __len__(self):
        return (len(self.indices_2d) + self.batch_size - 1) // self.batch_size + \
            (len(self.indices_3d) + self.batch_size - 1) // self.batch_size



"""
class AlternatingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indices_2d = [i for i in range(len(dataset)) if i % 2 == 0]  # Even indices (2D samples)
        self.indices_3d = [i for i in range(len(dataset)) if i % 2 == 1]  # Odd indices (3D samples)
        self.current_mode = "2D"  # Start with 2D batches
        self.switch_count = 0  # Track how many batches have been yielded in current mode
        self.switch_max = 4 * int(os.environ["SLURM_NNODES"])
    def __iter__(self):
        idx_2d, idx_3d = 0, 0
        while idx_2d < len(self.indices_2d) or idx_3d < len(self.indices_3d):
            if self.current_mode == "2D" and idx_2d < len(self.indices_2d):
                yield self.indices_2d[idx_2d : idx_2d + self.batch_size]
                idx_2d += self.batch_size
            elif self.current_mode == "3D" and idx_3d < len(self.indices_3d):
                yield self.indices_3d[idx_3d : idx_3d + self.batch_size]
                idx_3d += self.batch_size

            self.switch_count += 1
            if self.switch_count >= self.switch_max:  # Switch every 4 batches
                self.current_mode = "3D" if self.current_mode == "2D" else "2D"
                self.switch_count = 0

    def __len__(self):
        return (len(self.indices_2d) + self.batch_size - 1) // self.batch_size + \
            (len(self.indices_3d) + self.batch_size - 1) // self.batch_size

"""
##not care about this
"""
import random
from torch.utils.data import Sampler

class AlternatingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        # Separate indices for 2D and 3D samples.
        self.indices_2d = [i for i in range(len(dataset)) if i % 2 == 0]
        self.indices_3d = [i for i in range(len(dataset)) if i % 2 == 1]
        if self.shuffle:
            random.shuffle(self.indices_2d)
            random.shuffle(self.indices_3d)

    def __iter__(self):
        idx_2d = 0
        idx_3d = 0
        # For continuous sampling, we use an infinite loop.
        # (In practice, you may want to stop after a fixed number of batches.)
        while True:
            # If 2D indices are exhausted, restart them.
            if idx_2d >= len(self.indices_2d):
                idx_2d = 0
                if self.shuffle:
                    random.shuffle(self.indices_2d)
            # Yield one 2D batch.
            yield self.indices_2d[idx_2d : idx_2d + self.batch_size]
            idx_2d += self.batch_size

            # Yield four 3D batches.
            for _ in range(4):
                if idx_3d >= len(self.indices_3d):
                    idx_3d = 0
                    if self.shuffle:
                        random.shuffle(self.indices_3d)
                yield self.indices_3d[idx_3d : idx_3d + self.batch_size]
                idx_3d += self.batch_size

    def __len__(self):
        # Length is ambiguous for an infinite sampler.
        # For a fixed epoch, you could define an epoch size.
        # For example, if you want one epoch to be a complete pass over both sets:
        num_batches_2d = (len(self.indices_2d) + self.batch_size - 1) // self.batch_size
        # Here, however, we cycle both indices so that 3D is sampled four times per 2D batch.
        # One possible epoch length is:
        return num_batches_2d * 5

"""

def update_conv_weights(state_dict):
    updated_state_dict = {}
    for key, value in state_dict.items():
        # Update keys containing 'conv' with '.bias' or '.weight' at the end
        if "conv" in key and (key.endswith(".bias") or key.endswith(".weight")):
            if key.endswith(".bias"):
                new_key = key.replace(".bias", ".conv.bias")
            elif key.endswith(".weight"):
                new_key = key.replace(".weight", ".conv.weight")
        else:
            # Keep the key unchanged if it doesn't meet the criteria
            new_key = key
        updated_state_dict[new_key] = value
    return updated_state_dict

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,backend:native"
def cycle(dl):
    while True:
        for data in dl:
            yield data

class EntropyLossAnnealingScheduler():
    def __init__(self, value=0.1, scaling_factor=3.0, decay_steps=2000):
        self.decay_steps = decay_steps
        self.max_value = scaling_factor * value
        self.final_value = value

    def step(self, current_step):
        if current_step < self.decay_steps:
            current_value = (self.max_value - self.final_value) / self.decay_steps * (self.decay_steps - current_step) + self.final_value
        else:
            current_value = self.final_value
        return current_value


class CosineWarmupScheduler:
    def __init__(self, initial_value, final_value, warmup_steps, last_step=-1):
        """
        :param param: Parameter to apply the cosine warmup scheduler to.
        :param warmup_steps: Number of steps over which to perform the warmup.
        :param max_steps: Total number of steps for the entire schedule.
        :param initial_value: Initial value of the parameter at the start of warmup.
        :param final_value: Final value of the parameter at the end of warmup.
        :param last_step: The index of the last step. Default is -1.
        """
        self.value = initial_value
        self.warmup_steps = warmup_steps
        self.initial_value = initial_value
        self.final_value = final_value

    def step(self, current_step):
        if current_step <= self.warmup_steps:
            # Cosine warmup
            cos_inner = (math.pi * current_step) / (2 * self.warmup_steps)
            new_value = self.initial_value + (self.final_value - self.initial_value) * math.sin(cos_inner)
            self.value = new_value
            return self.value
        else:
            # Hold final value after warmup period
            self.value = self.final_value
            return self.value

    def get_last_value(self):
        return self.value


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class VideoTokenizerTrainer:

    def __init__(self, model_config, trainer_config):
        self.unpatcher = UnPatcher3D(patch_size=2, patch_method="haar")
        self.patcher = Patcher3D(patch_size=2, patch_method="haar")

        self.trainer_config = trainer_config
        self.model_config = model_config
        set_seed(self.model_config.seed)
        self.debug_only = False
        self.use_ema = self.trainer_config.ema.apply_ema
        self.use_gan = self.trainer_config.gan.use_gan
        self.quantizer_type = (self.model_config.model.quantize_model.quantizer_type).lower()

        grad_accum_plugin = GradientAccumulationPlugin(num_steps=self.trainer_config.optimizer.grad_accum_size, sync_each_batch=True, sync_with_dataloader=False)
        timeout = timedelta(days=2)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=False, broadcast_buffers=False)
        init_pg_kwargs = InitProcessGroupKwargs(timeout=timeout)

        self.accelerator = Accelerator(log_with="tensorboard", 
                                       project_dir=self.trainer_config.logging.tensorboard_dir,gradient_accumulation_plugin=grad_accum_plugin,
                                       kwargs_handlers=[init_pg_kwargs,ddp_kwargs])
        self.dtype = self.get_dtype(self.accelerator.mixed_precision)

        if self.model_config.architecture == "magvit2":
            from modeling.magvit_model import VisionTokenizer
        else:
            raise NotImplementedError
        if self.quantizer_type == "lfq":
            self.model = VisionTokenizer(model_config, 
                                         self.trainer_config.quantizer.commitment_cost, 
                                         self.trainer_config.quantizer.diversity_gamma,
                                         use_gan=self.trainer_config.gan.use_gan,
                                         use_lecam_ema=self.trainer_config.gan.use_lecam_ema,
                                         use_perceptual=self.trainer_config.perceptual.use_perceptual,
                                         perceptual_ckpt_path=self.trainer_config.perceptual.ckpt_path)
            
        elif self.quantizer_type == "vq":
            raise NotImplementedError
        
        if trainer_config.checkpointing.inflate_from_2d is True:
            self.model = self.inflate_2d_to_3d(self.model, trainer_config.checkpointing.inflate_ckpt_path)

        if self.trainer_config.perceptual.use_perceptual:
            self.model.perceptual_model.to(self.dtype)
        #self.model.perceptual_model
        self.codebook_size = self.model.codebook_size


        if self.trainer_config.modal == "video":
            #dataset = VideoDataset(meta_path=self.trainer_config.data.train_dir, image_size=self.trainer_config.data.spatial_size, num_frames=self.trainer_config.data.num_frames)
            #valid_dataset = VideoDataset(meta_path=self.trainer_config.data.valid_dir, image_size=self.trainer_config.data.spatial_size, num_frames=self.trainer_config.data.num_frames)
            path_temp = os.environ["TMPDIR"]
            path_temp = "/anvme/workspace/b180dc42-ct-rate/data_volumes/dataset"
            #dataset = CTReportDataset(data_folder="/anvme/workspace/b180dc42-ct-rate/data_volumes/dataset/train", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/train_reports.csv", batch_size = self.trainer_config.data.train_batch_size)
            dataset = CTReportDataset(data_folder="/anvme/workspace/b180dc42-ct-rate/data_volumes/dataset/train", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/train_reports.csv")

            valid_dataset =CTReportDatasetinfer(data_folder=path_temp + "/valid", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/validation_reports.csv", labels="/anvme/workspace/b180dc42-ct-rate/data_volumes/validation_labels.csv")


        elif self.trainer_config.modal == "image":
            #dataset = ImageDataset(meta_path=self.trainer_config.data.train_dir, image_size=self.trainer_config.data.spatial_size)
            #valid_dataset = ImageDataset(meta_path=self.trainer_config.data.valid_dir, image_size=self.trainer_config.data.spatial_size)
            #dataset = CTReportDataset(data_folder="/anvme/workspace/b180dc42-ct-rate/data_volumes/dataset/train", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/train_reports.csv", batch_size = self.trainer_config.data.train_batch_size)
            dataset = CTReportDataset(data_folder="/anvme/workspace/b180dc42-ct-rate/data_volumes/dataset/train", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/train_reports.csv")

            valid_dataset =CTReportDatasetinfer(data_folder=path_temp + "/valid", csv_file="/anvme/workspace/b180dc42-ct-rate/data_volumes/validation_reports.csv", labels="/anvme/workspace/b180dc42-ct-rate/data_volumes/validation_labels.csv")


        #sampler = AlternatingBatchSampler(dataset, self.trainer_config.data.train_batch_size)

        self.train_dataloader = DataLoader(dataset, shuffle = True, batch_size=1, pin_memory=True, num_workers=self.trainer_config.data.num_workers)
        self.valid_dataloader = DataLoader(valid_dataset, shuffle=False, drop_last=False, pin_memory=True, num_workers=self.trainer_config.data.num_workers, batch_size=self.trainer_config.data.valid_batch_size)


        self.num_epochs = self.trainer_config.data.num_epochs
        self.total_training_steps = self.trainer_config.data.num_epochs * len(self.train_dataloader) // (self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps)

        self.gen_optimizer = self.init_optimizer(self.model.tokenizer, lr_scale=self.trainer_config.lr_scheduler.gen_lr_scale)
        self.discri_optimizer =  self.init_optimizer(self.model.discriminator, lr_scale=self.trainer_config.lr_scheduler.disc_lr_scale)

        self.gen_lr_scheduler = self.init_scheduler(self.gen_optimizer)
        self.discri_lr_scheduler = self.init_scheduler(self.discri_optimizer)

        self.entropy_weight_scheduler = EntropyLossAnnealingScheduler(value=self.trainer_config.quantizer.entropy_loss_weight, scaling_factor=self.trainer_config.quantizer.entropy_loss_scale_factor, decay_steps=self.trainer_config.quantizer.entropy_loss_decay_steps)
        

        self.accelerator.init_trackers(self.trainer_config.exp_name)
        # self.trainer_config.logging.tensorboard_dir
        self.tracker = self.accelerator.get_tracker("tensorboard")

        accelerator_to_prepare = [self.train_dataloader, self.valid_dataloader,  self.model, self.gen_optimizer, self.discri_optimizer]


        self.model = self.model.to(torch.bfloat16)


        self.train_dataloader, self.valid_dataloader, self.model, self.gen_optimizer, self.discri_optimizer = self.accelerator.prepare(*accelerator_to_prepare)
        
        if self.use_ema:
            tokenizer = self.model.module.tokenizer if self.accelerator.num_processes > 1 else self.model.tokenizer
            self.ema_model = EMA(tokenizer, decay=self.trainer_config.ema.decay_rate)
            self.ema_model = self.ema_model.to(torch.bfloat16)
            self.ema_model.to(self.accelerator.device)
        
        # del accelerator_to_prepare
        self.accelerator.register_for_checkpointing(self.gen_lr_scheduler, self.discri_lr_scheduler)
        self.accelerator.register_save_state_pre_hook(self.save_state_pre_hook)
        self.accelerator.register_load_state_pre_hook(self.load_state_pre_hook)

        self.step = 0
        self.epoch = 0

        if self.trainer_config.checkpointing.pretrained is not None:
            if self.trainer_config.checkpointing.continue_training is True:
                self.accelerator.load_state(self.trainer_config.checkpointing.pretrained, strict=False)
                #self.gen_optimizer.state = {}
                #self.discri_optimizer.state = {}

                pretrained_root_dir = pathlib.Path(self.trainer_config.checkpointing.pretrained).parent
                #ema_state = torch.load(os.path.join(pretrained_root_dir, f"iter_{self.step}_ema.ckpt"))
                #self.ema_model.load_state_dict(ema_state)
                #del ema_state
                self.continue_training = True
            else:
                # reset 
                load_model(self.model.module if self.accelerator.num_processes > 1 else self.model, self.trainer_config.checkpointing.pretrained, strict=False)
                self.continue_training = False
            self.print_global_rank_0(f"successfully load from pretrained {self.trainer_config.checkpointing.pretrained}")

        else:
            self.continue_training = False

        self.ckpt_base_dir = os.path.join(self.trainer_config.io.ckpt_base_dir, self.trainer_config.exp_name)
        
        self.output_base_dir = os.path.join(self.trainer_config.io.output_base_dir, self.trainer_config.exp_name)
        if self.accelerator.process_index == 0:
            os.makedirs(self.ckpt_base_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_base_dir, "input"), exist_ok=True)
            os.makedirs(os.path.join(self.output_base_dir, "output"), exist_ok=True)

        self.codebook_tracker = set()
        gc.collect()
        torch.cuda.empty_cache()      

    @staticmethod
    def get_dtype(dtype_str: str):
        dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        return dtype_map[dtype_str]

    def print_global_rank_0(self, *message):
        if self.accelerator.process_index == 0:
            print(*message)
    @property
    def apply_gradient_penalty(self):
        return self.trainer_config.gan.apply_gradient_penalty and (self.step + 1) % self.trainer_config.gan.apply_gradient_penalty_every == 0

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @staticmethod
    def inflate_2d_to_3d(model: torch.nn.Module, ckpt_path):
        """
        Inflates Conv2D weights to Conv3D weights, fixes their names, and loads them into the model.

        Args:
            model (torch.nn.Module): The target model with 3D convolutions.
            ckpt_path (str): Path to the checkpoint containing 2D weights.

        Returns:
            torch.nn.Module: The model with the inflated weights loaded.
        """
        # Load the checkpoint
        state_dict = torch.load(ckpt_path)

        # Iterate through state_dict to find, inflate, and fix names of Conv2D weights
        inflated_state_dict = {}
        ignore_layers = [
            "decoder.conv_upsample_blocks.conv_upsample_0.conv1.weight",
            "decoder.conv_upsample_blocks.conv_upsample_0.conv1.bias",
            "decoder.conv_upsample_blocks.conv_upsample_1.conv1.weight",
            "decoder.conv_upsample_blocks.conv_upsample_1.conv1.bias",
            "decoder.conv_upsample_blocks.conv_upsample_2.conv1.weight",
            "decoder.conv_upsample_blocks.conv_upsample_2.conv1.bias",
            "decoder.conv_upsample_blocks.conv_upsample_3.conv1.weight",
            "decoder.conv_upsample_blocks.conv_upsample_3.conv1.bias"
        ]

        for key, value in state_dict.items():
            if key not in ignore_layers:
                new_key = key
                if "conv" in key:
                    # Fix names: Add ".conv" before ".bias" or ".weight"
                    if key.endswith(".bias"):
                        new_key = key.replace(".bias", ".conv.bias")
                    elif key.endswith(".weight"):
                        new_key = key.replace(".weight", ".conv.weight")
                    # Inflate Conv2D weights to Conv3D weights if applicable
                    if value.ndim == 4:  # Check if it's a Conv2D weight
                        value = value.unsqueeze(2)  # Add depth dimension
                        num = value.shape[-1]
                        value = value.repeat(1, 1, num, 1, 1)  # Inflate depth to 1

                # Add the processed weights to the new state dictionary
                inflated_state_dict[new_key] = value

        # Load the updated state dict into the model
        #try:
        model.tokenizer.load_state_dict(inflated_state_dict, strict=False)
        #print("Weights successfully inflated, renamed, and loaded.")
        #except RuntimeError as e:
        #    print(f"Failed to load state dict: {e}")

        return model

    def save_state_pre_hook(self, models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], input_dir: str):
        self.trainer_config.current_step = self.step
        self.trainer_config.current_micro_step = self.accelerator.step
        self.trainer_config.current_epoch = self.epoch

        if self.accelerator.process_index == 0:
            OmegaConf.save(self.trainer_config, os.path.join(self.ckpt_base_dir, f"iter_{self.step}", "trainer_config.yaml"))
            OmegaConf.save(self.model_config, os.path.join(self.ckpt_base_dir, f"iter_{self.step}", "model_config.yaml"))
        self.accelerator.wait_for_everyone()

    def load_state_pre_hook(self, models: list[torch.nn.Module], input_dir: str) -> None:
        trainer_config = OmegaConf.load(os.path.join(input_dir, "trainer_config.yaml"))
        model_config = OmegaConf.load(os.path.join(input_dir, "model_config.yaml"))
        self.accelerator.wait_for_everyone()
        self.step = trainer_config.pop("current_step")
        self.epoch = trainer_config.pop("current_epoch")
        self.accelerator.step = trainer_config.pop("current_micro_step")
    

    def train_step(self, batch):
        data, index_strs, data_mode = batch
        #print(data_mode, "data mode")
        #print(index_strs)
        data_mode = data_mode[0]
        if data_mode == "2d":
            #print("running 2d step")
            #print(data.shape)
            data = data.permute(0, 2, 1, 3, 4)
            batch = data.shape[0]
            num_slices = data.shape[1]
            #print(batch,"batch")
            #print(num_slices, "num_slices")
            data = data.reshape(batch * num_slices, 1, 512, 512).unsqueeze(2)

        #data = data[0, :, torch.randperm(data.size(2)), :, :].permute(1, 0, 2, 3)
        #data = data[0:8]
        #print(data.shape)
        c = data.shape[2]
        start_idx = random.randint(0, c - 105)

        #start_idx = random.randint(0, c - 5)
        data = data[:,:,start_idx : start_idx + 105,:,:]

        self.print_global_rank_0(f"RECORD: step {self.step}, indices: {list(map(int, index_strs))}")

        # ====
        with self.accelerator.accumulate(self.model):

            is_gradient_accum_boundary = (self.accelerator.step % (self.accelerator.gradient_accumulation_steps)== 0)

            quantizer_entropy_loss_weight = self.entropy_weight_scheduler.step(self.step)

            # **** run reconstruction ****
            loss_weights = {
                "recon_loss_weight": self.trainer_config.recon_loss_weight,
                "quantizer_entropy_loss_weight": quantizer_entropy_loss_weight,
                "quantizer_aux_loss_weight": self.trainer_config.quantizer.aux_loss_weight
            }
            recon_forward_s = time.perf_counter()
            recon_output, indices, recon_loss, quantize_loss, quantize_loss_breakdown = self.model.forward(
                forward_mode="reconstruction",
                data_mode = data_mode,
                input_data=data,
                calculate_loss=False,
                loss_weights=loss_weights,
                use_distributed_batch_entropy=self.trainer_config.quantizer.use_distributed_batch_entropy,
            )
            recon_forward_e = time.perf_counter()
            recon_forward_cost = recon_forward_e - recon_forward_s

            loss_breakdowns = [quantize_loss_breakdown]
            # **** run generator ****
            #self.print_global_rank_0("run generator!!!")
            loss_weights.update(
                {
                    "g_adversarial_loss_weight": self.trainer_config.gan["g_adversarial_loss_weight"],
                    "perceptual_loss_weight": self.trainer_config.perceptual["perceptual_loss_weight"],
                }
            )
            # ---- forward ----
            gen_forward_s = time.perf_counter()
            gen_loss, loss_breakdown = self.model.forward(
                                        forward_mode="generator",
                                        data_mode = data_mode,
                                        input_data=data, 
                                        recon_output=recon_output,
                                        recon_loss=recon_loss, 
                                        quantize_loss=quantize_loss,
                                        generator_loss_type=self.trainer_config.gan.generator_loss_type,
                                        loss_weights=loss_weights,
                                    )    
            gen_forward_e = time.perf_counter()
            gen_forward_cost = gen_forward_e - gen_forward_s
            loss_breakdowns.append(loss_breakdown)

            # ---- backward ----
            self.gen_optimizer.zero_grad()
            self.accelerator.backward(gen_loss)
            
            if is_gradient_accum_boundary:

                should_skip_current_step = (self.accelerator.optimizer_step_was_skipped or torch.isnan(gen_loss))
                if not should_skip_current_step:
                    if self.trainer_config.max_grad_norm is not None:
                        #self.print_global_rank_0(f"clip grad norm to {self.trainer_config.max_grad_norm}...")
                        self.accelerator.clip_grad_norm_(self.model.module.tokenizer.parameters(), self.trainer_config.max_grad_norm)
                    last_gen_lr = self.gen_lr_scheduler.get_last_lr()[0]
                    self.gen_optimizer.step()
                    self.gen_lr_scheduler.step()

                    if self.use_ema:
                        self.ema_model.update(self.model.module.tokenizer if self.accelerator.num_processes > 1 else self.model.tokenizer, step=self.step)
                else:
                    self.print_global_rank_0("[WARN] generator optimizer_step_was_skipped! ")
            gen_loss = gen_loss.detach()

            # **** run discriminator ****
            #self.print_global_rank_0("run discriminator!!!")
            loss_weights.update(
                {
                    "lecam_loss_weight": self.trainer_config.gan["lecam_loss_weight"],
                    "gradient_penalty_cost": self.trainer_config.gan["gradient_penalty_cost"],
                    "d_adversarial_loss_weight": self.trainer_config.gan["d_adversarial_loss_weight"]
                    }
            )
            recon_output = torch.clone(recon_output.detach()) # to save memory
            # ---- forward ----
            disc_forward_s = time.perf_counter()

            disc_loss, loss_breakdown = self.model.forward(
                forward_mode="discriminator",
                data_mode = data_mode,
                input_data=data,
                recon_output=recon_output,
                apply_gradient_penalty=self.apply_gradient_penalty,
                discriminator_loss_type=self.trainer_config.gan.discriminator_loss_type,
                loss_weights=loss_weights,
            )

            disc_forward_e = time.perf_counter()
            disc_forward_cost = disc_forward_e - disc_forward_s
            loss_breakdowns.append(loss_breakdown)

            # ---- backward ----
            self.discri_optimizer.zero_grad()

            self.accelerator.backward(disc_loss)
            if is_gradient_accum_boundary:
                should_skip_current_step = (self.accelerator.optimizer_step_was_skipped or torch.isnan(disc_loss))
                if not should_skip_current_step:
                    if self.trainer_config.max_grad_norm is not None:
                        #self.print_global_rank_0(f"clip grad norm to {self.trainer_config.max_grad_norm}...")
                        self.accelerator.clip_grad_norm_(self.model.module.discriminator.parameters(), self.trainer_config.max_grad_norm)

                    last_disc_lr = self.discri_lr_scheduler.get_last_lr()[0]
                    self.discri_optimizer.step()
                    self.discri_lr_scheduler.step()
                else:
                    self.print_global_rank_0("[WARN] discriminator optimizer_step_was_skipped! ")
            disc_loss = disc_loss.detach()

            # **** logging ****
            tracker_dict = dict()
            for loss_group in loss_breakdowns:
                loss_type = type(loss_group).__name__
                if loss_type == "QuantLossBreakdown":
                    tracker_prefix = "quantize/"
                elif loss_type == "GenLossBreakdown":
                    tracker_prefix = "generator/"
                elif loss_type == "DiscriLossBreakdown":
                    tracker_prefix = "discriminator/"
                elif loss_type == "EncDecLossBreakdown":
                    tracker_prefix = "encdec/"
                else:
                    raise NotImplementedError
                for field, value in zip(loss_group._fields, loss_group):
                    if value is not None:                        
                        avg_val = self.accelerator.gather_for_metrics(value).mean()
                        tracker_dict[tracker_prefix+field] = avg_val.item()
            self.tracker.log(
                tracker_dict, step=self.step
            )

            self.tracker.log({
                "quantize/entropy_loss_weight": quantizer_entropy_loss_weight}, step=self.step)
            if self.quantizer_type == "lfq":
                self.tracker.log({
                    "quantize/diversity_gamma": self.trainer_config.quantizer.diversity_gamma}, step=self.step)

            with torch.no_grad():
                tokens_per_batch = self.accelerator.gather_for_metrics(indices.detach().reshape(-1))
                unique_tokens_per_batch = tokens_per_batch.unique().tolist()
                self.tracker.log({"train/batch_unique_code_ratio": round(len(unique_tokens_per_batch) / min(self.codebook_size, len(tokens_per_batch.tolist())), 4)}, step=self.step)
                del tokens_per_batch, unique_tokens_per_batch

            if is_gradient_accum_boundary:
                self.step += 1
                if self.accelerator.process_index == 0:
                    self.time_e = time.perf_counter()
                    duration = round(self.time_e - self.time_s, 2)
                    self.accum_duration += duration
                    if self.step % 100 == 0:
                        self.print_global_rank_0(f"time consumed for 100 step: {self.accum_duration} seconds")
                    self.time_s = self.time_e
                    info = {"global_step": self.step, 
                            "micro_step": self.accelerator.step, 
                            "epoch": self.epoch, 
                            "duration": duration, 
                            "recon_forward_time": round(recon_forward_cost, 4),
                            "disciminator_forward_time": round(disc_forward_cost, 4),
                            "generator_forward_time": round(gen_forward_cost, 4)
                            }
                    if self.step % 10 == 0:
                        self.print_global_rank_0(f">> TRAINING INFO: {json.dumps(info)}")
                
                self.tracker.log({"train/generator_lr": last_gen_lr}, step=self.step)
                self.tracker.log({"train/discriminator_lr": last_disc_lr}, step=self.step)
                mem_stats = torch.cuda.memory_stats()
                self.tracker.log({
                                "memory/reserved-bytes": mem_stats["reserved_bytes.all.current"],
                                "memory/allocated-bytes": mem_stats["allocated_bytes.all.current"],
                                "memory/allocated-count": mem_stats["allocation.all.current"]
                                },
                                step=self.step)
                input_video = data

                is_global_step_update = True
            else:
                input_video = None
                is_global_step_update = False

        return is_global_step_update, input_video, recon_output


    def train(self):

        self.print_global_rank_0(f"using global batch size: {self.train_dataloader.batch_sampler.batch_size} x {self.accelerator.state.num_processes} x {self.trainer_config.optimizer.grad_accum_size}")
        self.model.train()
        if self.accelerator.process_index == 0:
            self.time_s = time.perf_counter()

        if self.continue_training:
            micro_steps_to_skip = (self.step * self.accelerator.gradient_accumulation_steps) % len(self.train_dataloader)

            skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, num_batches=micro_steps_to_skip)
            num_trained_epochs = self.epoch
            if micro_steps_to_skip == 0:
                num_trained_epochs += 1
            
        else:
            micro_steps_to_skip = 0
            num_trained_epochs = 0

        self.print_global_rank_0(f"num steps to skip {micro_steps_to_skip}")
        valid_data_iter = iter(cycle(self.valid_dataloader))

        

        for epoch in range(self.num_epochs):
            if epoch < num_trained_epochs:
                continue
            if micro_steps_to_skip > 0:
                self.print_global_rank_0(f"{micro_steps_to_skip} steps trained for epoch {epoch}, continue training for rest...")
                train_dataloader = skipped_dataloader
                micro_steps_to_skip = 0
            else:
                train_dataloader = self.train_dataloader
            self.epoch = epoch
            
            d_s = time.time()


            for batch in train_dataloader:

                d_e = time.time()
                #print(batch[0].shape)
                self.print_global_rank_0(f">> TIME: [data loading] {d_e - d_s}")
                with self.accelerator.autocast():
                    if self.step % 100 == 0:
                        self.accum_duration = 0
                    is_global_step_update, input_video, recon_output = self.train_step(batch)
                self.accelerator.wait_for_everyone()
                if is_global_step_update:
                    if self.step > 0 and self.step % self.trainer_config.logging.validate_every_step == 0:
                        self.valid_step(valid_data_iter=valid_data_iter)                        
                        self.model.train()
                    if self.step > 0 and self.step % self.trainer_config.checkpointing.checkpoint_every_step == 0:
                        self.save_ckpt(self.step)
                        self.accelerator.save_state(os.path.join(self.ckpt_base_dir, f"iter_{self.step}"))
                d_s = time.time()

    def write_recon_video(self, input_video, decoded_output, prefix=""):
        
        if self.accelerator.process_index == 0:
            write_video(os.path.join(self.output_base_dir, "input", f"input_step{self.step:05d}.mp4"), ((rearrange(input_video.detach().cpu().repeat(3,1,1,1), "c t h w -> t h w c")+1)*127.5).to(torch.uint8), fps=10)
            write_video(os.path.join(self.output_base_dir, "output", f"{prefix}output_step{self.step:05d}.mp4"), ((rearrange(decoded_output.detach().cpu().repeat(3,1,1,1), "c t h w -> t h w c").clamp(-1,1)+1) * 127.5).to(torch.uint8), fps=10)
        #self.accelerator.wait_for_everyone()

    def write_recon_image(self, input_image, decoded_output, prefix=""):
        
        if self.accelerator.process_index == 0:
            input_fname = os.path.join(self.output_base_dir, "input", f"input_step{self.step:05d}.png")
            output_fname = os.path.join(self.output_base_dir, "output", f"{prefix}output_step{self.step:05d}.png")
            #print(input_image.shape)
            write_png(((input_image.detach().cpu().repeat(3,1,1) + 1)*127.5).to(torch.uint8), input_fname, compression_level=0)
            write_png(((decoded_output.detach().cpu().repeat(3,1,1).clamp(-1,1) +1)*127.5).to(torch.uint8), output_fname, compression_level=0)
        #self.accelerator.wait_for_everyone()

    def valid_step(self, valid_data_iter):
        if self.accelerator.is_main_process:

            self.model.eval()
            #self.ema_model.eval()
            with torch.no_grad():
                eval_data, *_ = next(valid_data_iter)
                #eval_data = eval_data[0, :, torch.randperm(eval_data.size(2)), :, :].permute(1, 0, 2, 3)
                #eval_data = eval_data[60:61]
                c = eval_data.shape[2]
                start_idx = random.randint(0, c - 105)
                eval_data = eval_data[:,:,start_idx : start_idx + 105,:,:]
                #all_eval_data = self.accelerator.gather_for_metrics(eval_data)
                all_eval_data = eval_data
                if self.use_ema:
                    eval_data = eval_data
                    ema_recon_output, *_ = self.ema_model.module(
                        eval_data,
                        entropy_loss_weight=0.0)
                    #all_ema_recon_output = self.accelerator.gather_for_metrics(ema_recon_output)
                    all_ema_recon_output = ema_recon_output
                    #valid_ema_recon_loss = F.mse_loss(self.patcher(all_eval_data.cpu().float()), all_ema_recon_output.cpu().float())
                    valid_ema_recon_loss = F.l1_loss(all_eval_data.cpu().float(), all_ema_recon_output.cpu().float())
                    self.tracker.log({"val/ema_recon_loss": valid_ema_recon_loss.item()}, step=self.step)

                loss_weights = {
                    "recon_loss_weight": 0.0,
                    "quantizer_entropy_loss_weight": 0.0,
                    "quantizer_aux_loss_weight": 0.0
                }

                recon_output, *_ = self.model(
                    forward_mode="reconstruction",
                    data_mode = "3d",
                    input_data=eval_data,
                    calculate_loss=True,
                    loss_weights=loss_weights
                )
                #all_recon_output = self.accelerator.gather_for_metrics(recon_output)
                all_recon_output = recon_output
                #valid_recon_loss = F.mse_loss(self.patcher(all_eval_data.cpu().float()), all_recon_output.cpu().float())
                valid_recon_loss = F.l1_loss(all_eval_data.cpu().float(), all_recon_output.cpu().float())
                self.tracker.log({"val/recon_loss": valid_recon_loss.item()}, step=self.step)

            sample_index = random.randint(0, all_eval_data.shape[0]-1)
            if self.trainer_config.modal == "video":
                #save_out = self.unpatcher(all_recon_output[sample_index].unsqueeze(0).cpu())
                save_out = all_recon_output[sample_index].unsqueeze(0)
                self.write_recon_video(all_eval_data[sample_index].cpu(), save_out[0].cpu().float())
                if self.use_ema:
                    #ema_out = self.unpatcher(all_ema_recon_output[sample_index].unsqueeze(0).cpu())
                    ema_out = all_ema_recon_output[sample_index].unsqueeze(0)
                    self.write_recon_video(all_eval_data[sample_index].cpu(), ema_out[0].cpu().float(), prefix="ema_")

            elif self.trainer_config.modal == "image":
                self.write_recon_image(all_eval_data[sample_index].cpu(), all_recon_output[sample_index].cpu())
                if self.use_ema:
                    self.write_recon_image(all_eval_data[sample_index].cpu(), all_ema_recon_output[sample_index].cpu(), prefix="ema_")
            #del eval_data, recon_output, ema_recon_output, all_eval_data, all_recon_output, all_ema_recon_output
            del eval_data, recon_output, all_eval_data, all_recon_output

        self.accelerator.wait_for_everyone()
        # gc.collect()
        # torch.cuda.empty_cache()


    def init_optimizer(self, model, lr_scale: float):

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=self.trainer_config.lr_scheduler.learning_rate * lr_scale,
            betas=(self.trainer_config.optimizer.beta1, self.trainer_config.optimizer.beta2),
            weight_decay=0,
            eps=self.trainer_config.optimizer.epsilon,
            foreach=True
        )
        return optimizer

    def init_scheduler(self, optimizer):
        warmup_steps = self.trainer_config.lr_scheduler.warmup_steps
        lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_training_steps)
        return lr_scheduler


    def save_ckpt(self, step):
        if self.use_ema:
            self.accelerator.save(self.ema_model.state_dict(), os.path.join(self.ckpt_base_dir, f"iter_{step}_ema.ckpt"))
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.ckpt_base_dir, f"iter_{step}.ckpt"))
        self.accelerator.wait_for_everyone()

    def load_ckpt(self, step):
        torch.load(os.path.join(self.ckpt_base_dir, f"iter_{step}.ckpt"))
