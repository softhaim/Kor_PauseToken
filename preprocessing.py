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