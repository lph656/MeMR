"""
A cleaned and simplified configuration file for the medical continual learning project.
Unused arguments from the original classification task have been removed for clarity.
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # 核心参数：定义持续学习的任务顺序
    task_list: str = field(
        default='neike_waike_erke_fuchanke_nanke_zhongliuke',
        metadata={"help": "Task list for continual learning, order matters."}
    )
    
    # 核心参数：定义从训练集中划分多少作为验证集
    validation_split_percentage: float = field(
        default=0.1,
        metadata={"help": "The percentage of the train set to be used as a validation set."}
    )

    # 核心参数：序列和目标的最大长度
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    
    max_seq_length_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "A string of task-specific max sequence lengths, separated by underscores (e.g., '512_256_512'). "
            "If provided, this will override the global max_seq_length for each task."
        }
    )

    max_target_length: int = field(
        default=512,
        metadata={"help": "The maximum target sequence length for generation."}
    )


    # 功能性参数
    padding_strategy: str = field(
        default="longest", # 'longest' 通常比 'max_length' 更高效
        metadata={"help": "Padding strategy. Choices: ['max_length', 'longest', 'do_not_pad']"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    early_stop: bool = field(
        default=True, metadata={"help": "Use early stopping or not."}
    )
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={"help": "Patience for early stopping."}
    )
    learning_rate_list: Optional[str] = field(
        default=None,
        metadata={"help": "Use a different learning rate for each task, separated by '_'."}
    )
    max_train_batches_per_epoch: Optional[int] = field(
        default=None,
        metadata={"help": "Cap the number of training batches processed in each epoch for quick smoke runs."}
    )
    max_eval_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Cap the number of validation/test batches processed in non-final evaluation."}
    )
    max_final_test_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Cap the number of batches processed in the final all-task evaluation stage."}
    )
    
    # 保留 pad_to_max_length 以兼容旧的调用，但建议在脚本中不使用它
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Legacy argument. 'padding_strategy' is preferred."}
    )
    # 保留 add_dataset_name 以兼容旧的调用，但我们的新dataloader不使用它
    add_dataset_name: bool = field(
        default=False, metadata={"help": "Legacy argument. Not used in the current dataloader."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="chinese-alpaca-plus-7b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    meta_embeddings_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-encoded metadata embeddings (.pt file) for task key initialization and dynamic matching."}
    )
    # --- MPEFT 框架核心参数 (Mixture of PEFT modules) ---
    mpeft_enabled: bool = field(
        default=True,
        metadata={"help": "Enable 'mixture of peft modules' functionalities."}
    )
    continual_learning: bool = field(
        default=True, # 默认为 True，因为这就是项目的核心
        metadata={"help": "Run in the continual learning mode."}
    )
    query_encoder_type: str = field(
        default="avg_word_embed", # avg_word_embed 通常比 avg_all_embed 更鲁棒
        metadata={"help": "Embedding type for query encoder. Options: 'avg_all_embed', 'avg_word_embed'."}
    )
    matching_loss_v2: bool = field(
        default=True, # 默认开启，这是 MPEFT 的一个关键特性
        metadata={"help": "Enable query <-> key matching loss."}
    )
    matching_loss_coeff: float = field(
        default=1.0,
        metadata={"help": "Coefficient for the matching loss term."}
    )

    multi_peft_modules: bool = field(
        default=True,
        metadata={
            "help": "Multi-task learning with multiple prefix (for composition / concatenation / identification)"
        }
    )
    
    # --- LLaMA 特有的 LoRA 相关配置 (原框架保留的) ---
    disentangle_modules: bool = field(
        default=False,
        metadata={"help": "Used for per-task fine-tuning and EPI inference (an alternative CL strategy)."}
    )
    task_identify_epi: bool = field(
        default=False,
        metadata={"help": "Select module based on Gaussian distribution prototypes during inference (EPI strategy)."}
    )
    
    # --- 其他可选的框架参数 ---
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "The dropout probability used in the models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use."},
    )

@dataclass
class OSLArguments:
    """
    Arguments specific to Orthogonal Subspace Learning (OSL).
    """
    lamda_1: float = field(
        default=0.1,
        metadata={"help": "Weight for the orthogonalization loss (OSL) between current and previous LoRA parameters."}
    )
    lamda_2: float = field(
        default=0.01,
        metadata={"help": "Weight for the L2 regularization loss on current task's trainable LoRA parameters."}
    )
    orthogonal_threshold: float = field(
        default=0.1, # 新增参数：正交性阈值
        metadata={"help": "Threshold for orthogonalization loss. Only penalize if absolute dot product exceeds this value. (Soft Orthogonalization)"}
    )
    atr_enable_key_ortho: bool = field(
        default=False,
        metadata={"help": "Enable orthogonality regularization on task-representation vectors in the key encoder."}
    )
    atr_key_ortho_coeff: float = field(
        default=0.1,
        metadata={"help": "Weight for task-vector orthogonality regularization in ATR reviewer experiments."}
    )
    atr_key_ortho_threshold: float = field(
        default=0.1,
        metadata={"help": "Soft-threshold used by task-vector orthogonality regularization."}
    )
    atr_key_l2_lambda: float = field(
        default=0.0,
        metadata={"help": "L2 regularization strength on task-representation vectors in the key encoder."}
    )
    atr_key_ortho_mode: str = field(
        default="unnormalized_soft",
        metadata={"help": "Orthogonality mode for task vectors: unnormalized_soft, hard, cosine_soft."}
    )
    atr_variant_name: str = field(
        default="default",
        metadata={"help": "Human-readable experiment variant name for ATR reviewer studies."}
    )
    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the model base loss. Use 0.0 for regularizer-only diagnostic runs."}
    )
# 注意：get_args() 函数被移除了，因为它在 run_continual_causal_llama2.py 中没有被调用。
# 主脚本直接使用 HfArgumentParser，这更加标准。
