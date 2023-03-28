import torch
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AutoConfig
from datasets import load_dataset, DatasetDict
from tqdm.auto import tqdm
from datasets import load_dataset
from helpers import tokenize_dataset



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# define tokens 
bos = '<|endoftext|>'
eos = '<|EOS|>'
pad = '<|pad|>'
body = '<|body|>'
additional_special_tokens = [body]

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad,'sep_token': body}

# tadd tokes to tokenizer
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# the model config to which we add the special tokens
config = AutoConfig.from_pretrained('gpt2', 
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)

# the pre-trained model is loaded with the custom configuration
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)

# the model embedding is resized
model.resize_token_embeddings(len(tokenizer))


# Load dataset 
arxiv_dataset = (load_dataset(path="CShorten/ML-ArXiv-Papers", split='train').train_test_split(train_size=90000, test_size=10000))
arxiv_dataset = DatasetDict({
    'train': arxiv_dataset['train'],
    'validation': arxiv_dataset['test']})


arxiv_dataset = arxiv_dataset.map(lambda x: {"abstracttitle": ' '.join([bos, x['abstract'], body, x['title'], eos]).replace("\n"," ")})

# Tokenizing dataset
tokenized_dataset = arxiv_dataset.map(lambda x: tokenizer(x["abstracttitle"], padding="max_length", 
                     truncation=True, max_length=512))


# Prepare dataset for training
train_dataset = tokenized_dataset["train"].remove_columns(['Unnamed: 0.1', 'Unnamed: 0', 'title', 'abstract', 'abstracttitle'])
val_dataset = tokenized_dataset["validation"].remove_columns(['Unnamed: 0.1', 'Unnamed: 0', 'title', 'abstract', 'abstracttitle'])

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

model_path = './model_title_from_abstract'

training_args = TrainingArguments(
    output_dir=model_path,          
    num_train_epochs=2,              
    per_device_train_batch_size=5,  
    per_device_eval_batch_size=5,   
    warmup_steps=200,                
    weight_decay=0.01,               
    logging_dir=model_path,            
    prediction_loss_only=True,
    save_steps=10000 
)

trainer = Trainer(
    model=model,                         
    args=training_args,                 
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],        
    eval_dataset=tokenized_dataset['validation']       
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(model_path)

eval_results = trainer.evaluate()

print(eval_results)