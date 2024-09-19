import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM

from utils import set_seed, save_result_to_csv, check_if_already_evaluated, load_latest_checkpoint
from model import load_model_and_tokenizer
from preprocessing import preprocess_data_with_pause_before_prefix, preprocess_data_with_pause_after_prefix
from trainer import CustomTrainer
from evaluation import evaluate_model

import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=20, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=20, help="Evaluation batch size per device")
    args = parser.parse_args()

    model_name = "skt/kogpt2-base-v2"
    pause_token_counts = [0, 5, 10]
    results_files = {'preappend': 'experiment_results_preappend.csv', 'append': 'experiment_results_append.csv'}
    preprocess_functions = {'preappend': preprocess_data_with_pause_before_prefix, 'append': preprocess_data_with_pause_after_prefix}
    
    eval_result, training_time = None, None

    # 랜덤 시드 설정
    set_seed(42)

    # 기기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, pause_token = load_model_and_tokenizer(model_name, device)
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    
    korquad_dataset = load_dataset("KorQuAD/squad_kor_v1")
    klue_nli_dataset = load_dataset("klue", "nli")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args_template = {
        "output_dir": "./results",
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "save_steps": 100,
        "save_total_limit": 2,
        "learning_rate": 5e-5,
        "fp16": True, 
        "logging_dir": "./logs",
        "dataloader_num_workers": 4,
        "ddp_find_unused_parameters": False,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
    }

    for preprocess_type, preprocess_func in preprocess_functions.items():
        results_file = results_files[preprocess_type]

        for num_pause_tokens in pause_token_counts:
            training_args = TrainingArguments(**training_args_template)

            for dataset_name, train_dataset, validation_dataset, task_type in [
                ('KLUE NLI', 
                klue_nli_dataset['train'].map(lambda x: preprocess_func(x, tokenizer, 'klue_nli', num_pause_tokens), batched=True),
                klue_nli_dataset['validation'].map(lambda x: preprocess_func(x, tokenizer, 'klue_nli', num_pause_tokens), batched=True),
                'NLI'),
                ('KorQuAD', 
                korquad_dataset['train'].map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
                korquad_dataset['validation'].map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
                'QA'),
            ]:

                trainer_type_name = CustomTrainer.__name__
                print(f"Checking if {model_name} with {num_pause_tokens} pause tokens on {dataset_name} dataset using {trainer_type_name} with {preprocess_type} preprocessing is already evaluated...")

                if check_if_already_evaluated(results_file, dataset_name, model_name, num_pause_tokens, trainer_type_name):
                    print(f"Already evaluated. Skipping evaluation for {dataset_name} with {num_pause_tokens} pause tokens using {trainer_type_name} with {preprocess_type} preprocessing.")
                    continue

                print(f"Fine-tuning {model_name} with {num_pause_tokens} pause tokens on {dataset_name} dataset using {trainer_type_name} with {preprocess_type} preprocessing...")

                model_output_dir = f"./results/{dataset_name}_{trainer_type_name}_pause_{num_pause_tokens}_{preprocess_type}"
                training_args.output_dir = model_output_dir

                latest_checkpoint = load_latest_checkpoint(model_output_dir)
                if latest_checkpoint:
                    print(f"Loading model from latest checkpoint: {latest_checkpoint}")
                    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint).to(device)
                    
                    print(f"Evaluating model from checkpoint {latest_checkpoint} on {dataset_name} with {num_pause_tokens} pause tokens using {trainer_type_name} with {preprocess_type} preprocessing...")
                    trainer = CustomTrainer(
                        model=model,
                        args=training_args,
                        eval_dataset=validation_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        num_pause_tokens=num_pause_tokens
                    )
                    
                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=args.eval_batch_size)
                    print(f"{dataset_name} Evaluation Result with {trainer_type_name} with {preprocess_type} preprocessing:", eval_result)
                    
                elif os.path.exists(model_output_dir) and os.path.exists(os.path.join(model_output_dir, "config.json")):
                    print(f"Model already exists at {model_output_dir}. Skipping training.")
                    model = AutoModelForCausalLM.from_pretrained(model_output_dir).to(device)
                    
                    print(f"Evaluating model from checkpoint {latest_checkpoint} on {dataset_name} with {num_pause_tokens} pause tokens using {trainer_type_name} with {preprocess_type} preprocessing...")
                    trainer = CustomTrainer(
                        model=model,
                        args=training_args,
                        eval_dataset=validation_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        num_pause_tokens=num_pause_tokens
                    )
                    
                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=args.eval_batch_size)
                    print(f"{dataset_name} Evaluation Result with {trainer_type_name} with {preprocess_type} preprocessing:", eval_result)
                    
                else:
                    trainer = CustomTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=validation_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        num_pause_tokens=num_pause_tokens
                    )

                    start_time = time.time()
                    trainer.train()
                    end_time = time.time()
                    training_time = end_time - start_time
                    
                    trainer.save_model(model_output_dir)
                    print(f"Model saved to {model_output_dir}")

                    latest_checkpoint = load_latest_checkpoint(model_output_dir)
                    model = AutoModelForCausalLM.from_pretrained(latest_checkpoint).to(device)

                    print(f"Evaluating model from checkpoint {latest_checkpoint} on {dataset_name} with {num_pause_tokens} pause tokens using {trainer_type_name} with {preprocess_type} preprocessing...")
                    trainer = CustomTrainer(
                        model=model,
                        args=training_args,
                        eval_dataset=validation_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        num_pause_tokens=num_pause_tokens
                    )

                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=args.eval_batch_size)
                    print(f"{dataset_name} Evaluation Result with {trainer_type_name} with {preprocess_type} preprocessing:", eval_result)

                trainer.model.train()

                if training_time:
                    print("Training complete. Model saved.")
                else:
                    print("No training performed. Skipping model save.")
                    training_time = None

                result = {
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Pause Tokens': num_pause_tokens,
                    'Trainer': trainer_type_name,
                    'Preprocessing': preprocess_type,
                    'Training Time (s)': training_time
                }

                result.update(eval_result)

                save_result_to_csv(result, results_file)
                
                del model
                torch.cuda.empty_cache()
                
                model, tokenizer, pause_token = load_model_and_tokenizer(model_name, device)

if __name__ == "__main__":
    main()
