import os
import torch
from copy import deepcopy     
from mpeft.utils.save_and_load import get_peft_model_state_dict
import traceback

class EarlyStopping:
    def __init__(self, save_path, logger, patience=5, verbose=False, delta=0):
        self.save_path = save_path
        self.logger = logger
        self.patience = patience
        # 是否打印详细日志
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_max = 0.
        self.delta = delta
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    # 定义调用方法，检查早停条件
    def __call__(self, score, model, task, save=True):

        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(score, model, task)
            
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(score, model, task)
            self.counter = 0
        
        return self.early_stop


    def save_checkpoint(self, score, model, task):
        # 修改7
        """
        if self.verbose:
            self.logger.info(f'Validation score increased ({self.val_score_max:.4f} --> {score:.4f}).  Saving model ...')

        path = os.path.join(self.save_path, f'best_checkpoint_{task}.pth')
        try:
            lora_params = get_peft_model_state_dict(model, adapter_name=task)
            # torch.save(lora_params, path)
            self.best_model = deepcopy(lora_params)
        except:
            torch.save(model.state_dict(), path)
        
        self.val_score_max = score
        """
        if self.verbose:
            self.logger.info(f'Validation score improved ({self.val_score_max:.4f} --> {score:.4f}). Saving PEFT model state dict...')
        try:
            # 尝试提取LoRA参数
            self.logger.info(f"Attempting to get PEFT model state dict for task '{task}'...")
            lora_params = get_peft_model_state_dict(model, adapter_name=task, save_embedding_layers=False)

            if not lora_params:
                 self.logger.warning(f"get_peft_model_state_dict for task '{task}' returned an empty dictionary. PEFT parameters might be missing or not found.")
                 self.best_model = None
            else:
                 self.best_model = deepcopy(lora_params)
                 self.logger.info(f"Successfully extracted and stored PEFT state dict for task '{task}' in memory.")

        # 捕获提取LoRA参数的异常
        except Exception as e:
            self.logger.error(f"!!! Failed to get PEFT model state dict for task '{task}' !!!")
            self.logger.error(f"Error Type: {type(e).__name__}")
            self.logger.error(f"Error Message: {e}")
            error_traceback = traceback.format_exc()
            self.logger.error(f"Traceback:\n{error_traceback}")
            self.best_model = None
            self.logger.warning(f"Skipping update of best_model for task '{task}' due to the error above.")
        self.val_score_max = score


    def reinit(self):
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_max = 0.
        