from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from dataHelper import get_dataset
import logging
import evaluate
import wandb
import numpy as np
from typing import List
import torch
import random
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
from adapters import AdapterType, AdapterTrainer, AdapterConfigBase, init

'''
    Initialize logging, seed, argparse...
'''
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class DataTrainingArguments(TrainingArguments):
    learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for Adam."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per device during training."})
    output_dir: str = field(default="./output", 
                            metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    report_to: List[str] = field(default_factory=lambda: [], 
                                 metadata={"help": "The list of integrations to report the logs to."})
    num_train_step: int = field(default=500, metadata={"help": "Number of update steps between two evaluations if evaluation_strategy='steps'."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy to adopt during training."}) 
    eval_steps: int = field(default=100, metadata={"help": "Number of update steps between two evaluations if evaluation_strategy='steps'."})
    logging_dir: str = field(default="./logs", metadata={"help": "The directory where logs are stored."})
    logging_steps: int = field(default=10, metadata={"help": "Number of update steps between two logs if evaluation_strategy='steps'."})

@dataclass
class MetaArguments:
    dataset_names: List[str] = field(default_factory=lambda: 
                                    ['rotten_tomatoes' ,'ag_news' ,'go_emotions', 'dair-ai/emotion'],
                                    metadata={"help": "The name of the dataset(s) to use."})
    dbg: bool = field(default=True, metadata={"help": "Whether to run in debug mode."})
    rand_seed: int = field(default=2023, metadata={"help": "The seed for reproducibility."})
    method: str = field(default='none', metadata={"help": "The method to use, none, lora or adapter."})
    lora_rank: int = field(default=4)
    lora_alpha: int = field(default=32)
    lora_dropout: int = field(default=0.1)

@dataclass
class ModelArguments:
    model_name: str = field(default="microsoft/deberta-v3-base", metadata={"help": "Path to pre-trained model or shortcut name."})
    model_path: str = field(default="microsoft/deberta-v3-base", metadata={"help": "Path to pre-trained model or shortcut name."})

# Create the HfArgumentParser with the defined data classes
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MetaArguments))

# Parse the command line arguments into data class instances
model_args, training_args, meta_args = parser.parse_args_into_dataclasses()


setup_seed(meta_args.rand_seed)

'''
    Load datasets
'''
# Load and preprocess the datasets
dataset = get_dataset(meta_args.dataset_names)

'''
    Load models
'''
# Load model configuration, tokenizer, and model
num_labels= max(dataset['test']['label']) + 1
config = AutoConfig.from_pretrained(model_args.model_path,
                                    num_labels= num_labels)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_path, 
    ignore_mismatched_sizes=True,
    config=config
)
if meta_args.method == 'lora':
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=meta_args.lora_rank,
        lora_alpha=meta_args.lora_alpha,
        lora_dropout=meta_args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("-------------- USE LORA --------------")
elif meta_args.method == 'adapter':
    adapter_config = AdapterConfigBase.load("pfeiffer", non_linearity="gelu", reduction_factor=16)
    init(model)
    model.add_adapter("adapter", config=adapter_config)
    model.train_adapter("adapter")
    
    print("-------------- USE ADAPTER --------------")
else:
    print("-------------- USE NONE --------------")

'''
    Process datasets and build up data collator
'''
# Tokenize all texts and align the labels with them.
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=False, truncation=True)

# Process the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Define a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metric
f1_metric = evaluate.load('f1')
accuracy_metric = evaluate.load('accuracy')

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    micro_f1 = f1_metric.compute(predictions=predictions, references=labels, average='micro')['f1']
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')['f1']
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'accuracy': accuracy
    }
'''
    Initialize Trainer
'''

if meta_args.dbg:
    print("-------------- USE DEBUG MODE --------------")
    training_args.report_to = []
else:
    training_args.report_to = ["wandb"]
    wandb.init(project="Final project", 
               config=vars(training_args), 
               entity="vectorzhao",
            name=f"{model_args.model_name}_lr_{training_args.learning_rate}_bs_{training_args.per_device_train_batch_size}_epos_{training_args.num_train_epochs}_seed_{meta_args.rand_seed}_method_{meta_args.method}_data_{meta_args.dataset_names}_steps_{training_args.num_train_step}")

training_args.eval_steps = len(tokenized_datasets['train']) // (10 * training_args.per_device_train_batch_size)
training_args.max_steps = training_args.num_train_step
# training_args.set_lr_scheduler(name="cosine")
# training_args.eval_steps = 10

if meta_args != 'adapter':
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
else:
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


'''
    Training
'''
# Train the model
train_result = trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

'''
    Additional Code for Testing/Validation
'''

# Save model, tokenizer and training arguments
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)