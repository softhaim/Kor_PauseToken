# 모델 학습 및 평가 스크립트

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import numpy as np
import os
import csv
import time
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import glob

# 토크나이저 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 랜덤 시드 설정
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 기기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Pause token 정의 및 추가
    pause_token = "<pause>"
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'additional_special_tokens': [pause_token]})
    
    # 토크나이저에 pause_token 속성 추가
    tokenizer.pause_token = pause_token
    
    # 모델의 토큰 임베딩 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    return model, tokenizer, pause_token

# 한국어 접두사 목록
korean_prefixes = [
    '가', '가시', '강', '개', '건', '겉', '겹', '경', '고', '공', '과', '군', '귀', '극', '급',
    '난', '날', '남', '내', '냉', '노', '농', '다', '단', '담', '당', '대', '덧', '데', '도', '도래', '독', '돌', '둘', '뒤', '드', '들', '들이', '등', '떡',
    '막', '맏', '말', '맞', '맨', '맹', '먹', '명', '몰', '무', '미', '민', '반', '밭', '백', '벌', '범', '복', '본', '부', '불', '비', '빗',
    '살', '새', '샛', '생', '서', '선', '설', '소', '속', '쇠', '수', '숫', '시', '신', '실', '싯', '아', '알', '암', '양', '얼', '여', '역', '연', '엿',
    '온', '올', '왕', '외', '요', '웃', '원', '유', '잔', '잡', '장', '재', '저', '제', '조', '종', '준', '줄', '중', '진', '짓', '짝', '쪽', 
    '차', '찰', '참', '처', '초', '총', '최', '치', '친', '탈', '토', '통', '풋', '피', '한', '함', '핫', '항', '해', '헛', '호', '홀', '홑', '휘',
    '늦', '되', '메', '싯', '엇', '찰', '핫', '햇'
]

# 전처리 함수: 접두사 앞에 pause token 추가 (preappend)
def preprocess_data_with_pause_before_prefix(examples, tokenizer, task, num_pause_tokens):
    if task == 'korquad':
        inputs = [q + tokenizer.eos_token + a for q, a in zip(examples["question"], examples["context"])]
    elif task == 'klue_nli':
        inputs = [premise + tokenizer.eos_token + hypothesis for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]
        labels = examples['label']
    elif task == 'nsmc':
        inputs = examples["document"]
        labels = examples["label"]

    processed_inputs = []
    for input_text in inputs:
        tokens = input_text.split()
        new_tokens = []

        for token in tokens:
            prefix_found = False
            for prefix in korean_prefixes:
                if token.startswith(prefix):  # 접두사와 일치한다면
                    # 접두사 앞에 pause 토큰 추가
                    new_token = ''.join([tokenizer.additional_special_tokens[0]] * num_pause_tokens) + token
                    new_tokens.append(new_token.strip())  # 공백 없이 결합된 토큰 추가
                    prefix_found = True
                    break

            if not prefix_found:
                new_tokens.append(token.strip()) 

        processed_input_text = ' '.join(new_tokens)
        processed_inputs.append(processed_input_text)

    model_inputs = tokenizer(processed_inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    if task != 'korquad':
        model_inputs["labels"] = labels
        
    # print("=== Tokenization Results ===")
    # for i, input_text in enumerate(processed_inputs):
    #     print(f"Original Text {i+1}: {input_text}")
    #     print(f"Tokenized IDs {i+1}: {model_inputs['input_ids'][i].tolist()}")
    #     print(f"Decoded Tokens {i+1}: {tokenizer.decode(model_inputs['input_ids'][i])}")
    #     print("-" * 50)

    return model_inputs

# 전처리 함수: 접두사 뒤에 pause token 추가 (append)
def preprocess_data_with_pause_after_prefix(examples, tokenizer, task, num_pause_tokens):
    if task == 'korquad':
        inputs = [q + tokenizer.eos_token + a for q, a in zip(examples["question"], examples["context"])]
    elif task == 'klue_nli':
        inputs = [premise + tokenizer.eos_token + hypothesis for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])]
        labels = examples['label']
    elif task == 'nsmc':
        inputs = examples["document"]
        labels = examples["label"]

    processed_inputs = []
    for input_text in inputs:
        tokens = input_text.split()
        new_tokens = []

        for token in tokens:
            prefix_found = False
            for prefix in korean_prefixes:
                if token.startswith(prefix):  # 접두사와 일치한다면
                    # 접두사와 나머지 부분을 공백 없이 결합
                    rest_of_token = token[len(prefix):].strip()
                    new_token = prefix + ''.join([tokenizer.additional_special_tokens[0]] * num_pause_tokens) + rest_of_token
                    new_tokens.append(new_token.strip())  
                    prefix_found = True
                    break

            if not prefix_found:
                new_tokens.append(token.strip()) 

        processed_input_text = ' '.join(new_tokens)
        processed_inputs.append(processed_input_text)

    model_inputs = tokenizer(processed_inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    if task != 'korquad':
        model_inputs["labels"] = labels

    return model_inputs

class CustomTrainer(Trainer):
    def __init__(self, *args, num_pause_tokens=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_pause_tokens = num_pause_tokens
        self.use_fp16 = self.args.fp16

    def compute_loss(self, model, inputs, return_outputs=False):
        pause_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])
        input_ids = inputs.get("input_ids").to(self.args.device)
        attention_mask = inputs.get("attention_mask").to(self.args.device)

        if self.use_fp16:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        all_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        ignore_output_mask = (shift_labels != pause_token_id).float()
        masked_losses = all_losses * ignore_output_mask  

        total_loss = 0.0
        for batch_idx in range(shift_labels.size(0)):
            pause_token_indices = (shift_labels[batch_idx] == pause_token_id).nonzero(as_tuple=True)[0]

            if len(pause_token_indices) > 0:
                initial_loss = masked_losses[batch_idx, :pause_token_indices[0].item()].sum()
                total_loss += initial_loss

                previous_idx = pause_token_indices[0].item()  

                for i in range(0, len(pause_token_indices) - self.num_pause_tokens, self.num_pause_tokens):
                    pause_start = pause_token_indices[i].item()
                    next_start = pause_token_indices[i + self.num_pause_tokens].item() if (i + self.num_pause_tokens) < len(pause_token_indices) else shift_labels.size(1)

                    if next_start - 1 < shift_labels.size(1):
                        for _ in range(self.num_pause_tokens):
                            connected_loss = masked_losses[batch_idx, :next_start].sum()
                            total_loss += connected_loss
                        previous_idx = next_start - 1

                if previous_idx < shift_labels.size(1):
                    post_pause_loss = masked_losses[batch_idx, previous_idx:].sum()
                    total_loss += post_pause_loss
            else:
                total_loss += masked_losses[batch_idx].sum()

        loss = total_loss / ignore_output_mask.sum()

        adjusted_loss = loss / (1 + self.num_pause_tokens)
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

def evaluate_model(trainer, eval_dataset, task_type, pause_token_id, batch_size=8):
    trainer.model.eval()
    total_size = len(eval_dataset)
    exact_match_total = 0
    accuracy_total = 0
    f1_total = 0
    total_samples = 0

    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        batch = eval_dataset.select(range(start_idx, end_idx))

        with torch.no_grad():
            outputs = trainer.predict(batch)
            predictions = outputs.predictions
            labels = outputs.label_ids

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        predictions = np.argmax(predictions, axis=-1)
        labels = labels[:, 1:]
        predictions = predictions[:, :-1]

        min_length = min(predictions.shape[1], labels.shape[1])
        labels = labels[:, :min_length]
        predictions = predictions[:, :min_length]
        
        mask = (labels != -100) & (labels != pause_token_id)
        labels = labels[mask]
        predictions = predictions[mask]

        decoded_labels = trainer.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = trainer.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        print(f"Debug - Cleaned Labels Sample: {decoded_labels[:30]}")  # 첫 두 샘플만 출력
        print(f"Debug - Cleaned Predictions Sample: {decoded_predictions[:30]}")  # 첫 두 샘플만 출력

        if task_type == 'QA':
            exact_match = np.mean([int(np.array_equal(a, p)) for a, p in zip(labels, predictions)])
            f1 = f1_score(labels, predictions, average='weighted')
            exact_match_total += exact_match * len(labels)
            f1_total += f1 * len(labels)

        elif task_type in ['NLI', 'Sentiment']:
            accuracy = accuracy_score(labels.flatten(), predictions.flatten())
            f1 = f1_score(labels.flatten(), predictions.flatten(), average='weighted')
            accuracy_total += accuracy * len(labels)
            f1_total += f1 * len(labels)

        total_samples += len(labels)

    if task_type == 'QA':
        exact_match_final = exact_match_total / total_samples
        f1_final = f1_total / total_samples
        return {'exact_match': exact_match_final, 'f1': f1_final}

    elif task_type in ['NLI', 'Sentiment']:
        accuracy_final = accuracy_total / total_samples
        f1_final = f1_total / total_samples
        return {'accuracy': accuracy_final, 'f1': f1_final}

def save_result_to_csv(result, filename):
    file_exists = os.path.isfile(filename)
    
    # 'Preprocessing' 필드를 추가하여 CSV 필드명 확장
    fieldnames = ['Dataset', 'Model', 'Pause Tokens', 'Trainer', 'Preprocessing', 'Evaluation Result', 'Training Time (s)', 
                  'exact_match', 'f1', 'accuracy']

    with open(filename, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not file_exists or os.stat(filename).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)

def check_if_already_evaluated(results_file, dataset_name, model_name, num_pause_tokens, trainer_type):
    if not os.path.exists(results_file) or os.stat(results_file).st_size == 0:
        return False

    results_df = pd.read_csv(results_file)
    if 'Trainer' not in results_df.columns:
        return False

    already_evaluated = ((results_df['Dataset'] == dataset_name) &
                         (results_df['Model'] == model_name) &
                         (results_df['Pause Tokens'] == num_pause_tokens) &
                         (results_df['Trainer'] == trainer_type)).any()
    
    return already_evaluated

def load_latest_checkpoint(model_output_dir):
    checkpoint_dirs = glob.glob(os.path.join(model_output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
    return latest_checkpoint_dir

if __name__ == "__main__":
    model_name = "skt/kogpt2-base-v2"
    pause_token_counts = [0, 5, 10]
    results_files = {'preappend': 'experiment_results_preappend_test.csv', 'append': 'experiment_results_append_test.csv'}
    preprocess_functions = {'preappend': preprocess_data_with_pause_before_prefix, 'append': preprocess_data_with_pause_after_prefix}
    
    eval_result, training_time = None, None

    model, tokenizer, pause_token = load_model_and_tokenizer(model_name)
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    
    korquad_dataset = load_dataset("KorQuAD/squad_kor_v1")
    klue_nli_dataset = load_dataset("klue", "nli")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args_template = {
        "output_dir": "./results",
        "overwrite_output_dir": True,
        "num_train_epochs": 30,
        "per_device_train_batch_size": 20,
        "per_device_eval_batch_size": 20,
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
                # ('KLUE NLI', 
                # klue_nli_dataset['train'].map(lambda x: preprocess_func(x, tokenizer, 'klue_nli', num_pause_tokens), batched=True),
                # klue_nli_dataset['validation'].map(lambda x: preprocess_func(x, tokenizer, 'klue_nli', num_pause_tokens), batched=True),
                # 'NLI'),
                # ('KorQuAD', 
                # korquad_dataset['train'].map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
                # korquad_dataset['validation'].map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
                # 'QA'),

                ('KLUE NLI', 
                klue_nli_dataset['train'].select(range(20)).map(lambda x: preprocess_func(x, tokenizer, 'klue_nli',  num_pause_tokens), batched=True),
                klue_nli_dataset['train'].select(range(10)).map(lambda x: preprocess_func(x, tokenizer, 'klue_nli', num_pause_tokens), batched=True),
                'NLI'),
                ('KorQuAD', 
                korquad_dataset['train'].select(range(20)).map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
                korquad_dataset['train'].select(range(10)).map(lambda x: preprocess_func(x, tokenizer, 'korquad', num_pause_tokens), batched=True),
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
                    
                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=64)
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
                    
                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=64)
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

                    eval_result = evaluate_model(trainer, validation_dataset, task_type, pause_token_id, batch_size=64)
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
                
                model, tokenizer, pause_token = load_model_and_tokenizer(model_name)
