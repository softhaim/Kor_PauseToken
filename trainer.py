from transformers import Trainer
import torch

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