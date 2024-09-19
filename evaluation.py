from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

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