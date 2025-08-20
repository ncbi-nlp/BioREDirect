from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from random import random
from random import shuffle
import os

class HiddenTokenPooler(nn.Module):
    def __init__(self, bert_model):
        super(HiddenTokenPooler, self).__init__()
        # Initialize with the weights and biases from the BERT model's pooler
        self.dense = nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size)
        self.dense.weight.data = bert_model.pooler.dense.weight.data.clone()
        self.dense.bias.data = bert_model.pooler.dense.bias.data.clone()
        
        self.activation = nn.Tanh()

    def forward(self, hidden_states, token_index=0):
        if hidden_states.dim() == 4:
                
            # hidden_states: (B, N, T, H)
            B, N, T, H = hidden_states.shape

            if isinstance(token_index, int):
                token_index = torch.full((B,), token_index, dtype=torch.long, device=hidden_states.device)
            elif isinstance(token_index, list):
                token_index = torch.tensor(token_index, dtype=torch.long, device=hidden_states.device)
            elif token_index.dim() == 2 and token_index.size(1) == 1:
                token_index = token_index.squeeze(1)

            token_index = token_index.unsqueeze(1).expand(B, N)  # (B, N)

            batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(B, N)  # (B, N)
            sent_idx  = torch.arange(N, device=hidden_states.device).unsqueeze(0).expand(B, N)  # (B, N)

            token_hidden_state = hidden_states[batch_idx, sent_idx, token_index]  # (B, N, H)
        else:
            token_hidden_state = hidden_states[:, token_index, :]

        # dense + tanh
        dense_output = self.dense(token_hidden_state)  # (B, N, H)
        pooled_output = self.activation(dense_output)  # (B, N, H)
        return pooled_output

class BioREDirect(nn.Module):

    def __init__(self, 
                 in_bert_model         = None,  
                 label_to_id           = None,
                 relation_label_to_id  = None,
                 novelty_label_to_id   = None,
                 direction_label_to_id = None,
                 soft_prompt_len       = 0,
                 num_soft_prompt       = 1,
                 hidden_size           = 768,
                 use_single_chunk      = False):
        
        super(BioREDirect, self).__init__()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_model  = BertModel.from_pretrained(in_bert_model).to(self.device)
        self.hidden_size = self.bert_model.config.hidden_size
        
        self.soft_prompt = torch.nn.Embedding(num_soft_prompt, 
                                              soft_prompt_len * hidden_size).to(self.device)
        
        # Initialize soft prompts
        self.soft_prompt.weight.data.normal_(mean = 0.0, 
                                             std  = self.bert_model.config.initializer_range)

        self.rel_token_poolor = HiddenTokenPooler(self.bert_model)
        self.nov_token_poolor = HiddenTokenPooler(self.bert_model) if len(novelty_label_to_id) > 0 else None
        self.dir_token_poolor = HiddenTokenPooler(self.bert_model) if len(direction_label_to_id) > 0 else None

        self.rel_token_classifier = nn.Linear(self.hidden_size, len(relation_label_to_id))
        self.nov_token_classifier = nn.Linear(self.hidden_size, len(novelty_label_to_id)) if len(novelty_label_to_id) > 0 else None
        self.dir_token_classifier = nn.Linear(self.hidden_size, len(direction_label_to_id)) if len(direction_label_to_id) > 0 else None

        self.rel_thresholds = nn.Parameter(torch.full((len(relation_label_to_id),), 0.5))
        self.nov_thresholds = nn.Parameter(torch.full((len(novelty_label_to_id),), 0.5) if len(novelty_label_to_id) > 0 else None)
        self.dir_thresholds = nn.Parameter(torch.full((len(direction_label_to_id),), 0.5) if len(direction_label_to_id) > 0 else None)

        self.soft_prompt_len       = soft_prompt_len
        self.num_soft_prompt       = num_soft_prompt
        self.label_to_id           = label_to_id  # Add the label_to_id mapping as an attribute
        self.relation_label_to_id  = relation_label_to_id
        self.novelty_label_to_id   = novelty_label_to_id
        self.direction_label_to_id = direction_label_to_id
        self.hidden_size           = hidden_size
        self.use_single_chunk      = use_single_chunk

    def get_ne_reps(self, 
                    sequence_output, 
                    entity_indices):
        
        all_ne_reps = []

        for batch_idx, entity_index in enumerate(entity_indices):
            ne_reps = []
            for index in entity_index:
                start_token_idx, current_end_index = index[0], index[-1]
                # Update the sentence index if the token is a separator token
                for i in range(start_token_idx, current_end_index+1):
                    ne_reps.append(sequence_output[batch_idx][i])
            if len(ne_reps) == 0:
                all_ne_reps.append(sequence_output[batch_idx][0])
            else:
                ne_rep_tensor = torch.stack(ne_reps)
                #max_value, _ = torch.max(ne_rep_tensor, dim=0)
                attention_weights = torch.softmax(torch.matmul(ne_rep_tensor, ne_rep_tensor.transpose(-1, -2)), dim=-1)            
                ne_rep_tensor = torch.matmul(attention_weights, ne_rep_tensor)
                all_ne_reps.append(ne_rep_tensor[0])
        return all_ne_reps
    
    def get_token_reps(self, 
                       sequence_output, 
                       token_indices):
        
        all_token_reps = []

        for batch_idx, _token_indices in enumerate(token_indices):
            token_reps = []
            for index in _token_indices:
                # Update the sentence index if the token is a separator token
                token_reps.append(sequence_output[batch_idx][index])
            if len(token_reps) == 0:
                #all_ne_reps.append(torch.tensor(sequence_output[batch_idx][0]))
                all_token_reps.append(sequence_output[batch_idx][0])
            else:
                token_rep_tensor = torch.stack(token_reps)
                attention_weights = torch.softmax(torch.matmul(token_rep_tensor, token_rep_tensor.transpose(-1, -2)), dim=-1)            
                token_rep_tensor = torch.matmul(attention_weights, token_rep_tensor)
                all_token_reps.append(token_rep_tensor[0])
        return all_token_reps
     
    def get_special_token_reps(self, 
                               sequence_output, 
                               special_token_index):
        
        all_token_reps = []
        for batch_idx, _sequence_output in enumerate(sequence_output):
            all_token_reps.append(_sequence_output[special_token_index].clone().detach())
        return all_token_reps

    def forward(self, 
                input_ids,
                attention_mask        = None,
                pair_prompt_ids       = None,
                relation_token_index  = None,
                direction_token_index = None,
                novelty_token_index   = None):
        
        if self.use_single_chunk:
            rel_token_logits, nov_token_logits, dir_token_logits = self.__single_chunk_forward(
                input_ids, 
                attention_mask, 
                pair_prompt_ids, 
                relation_token_index, 
                direction_token_index, 
                novelty_token_index)
        else:
            rel_token_logits, nov_token_logits, dir_token_logits = self.__multi_chunk_forward(
                input_ids, 
                attention_mask, 
                pair_prompt_ids, 
                relation_token_index, 
                direction_token_index, 
                novelty_token_index)
        
        return rel_token_logits, nov_token_logits, dir_token_logits

    def __single_chunk_forward(self, 
                               input_ids, 
                               attention_mask        = None, 
                               pair_prompt_ids       = None,
                               relation_token_index  = None,
                               direction_token_index = None,
                               novelty_token_index   = None):
        
        batch_size = len(input_ids)
        rel_token_logits = []
        nov_token_logits = []
        dir_token_logits = []
        for b in range(batch_size):
            _input_ids             = input_ids[b]
            _attention_mask        = attention_mask[b]
            #_entity1_indices       = entity1_indices[b]
            #_entity2_indices       = entity2_indices[b]
            #_entity1_sent_ids      = entity1_sent_ids[b]
            #_entity2_sent_ids      = entity2_sent_ids[b]
            _pair_prompt_ids       = pair_prompt_ids[b]
            #_sent_ids              = sent_ids[b]
            _relation_token_index  = relation_token_index[b]
            _direction_token_index = direction_token_index[b]
            _novelty_token_index   = novelty_token_index[b]
            
            input_embeddings = self.bert_model.embeddings(_input_ids)
            # Generate soft prompt embeddings
            if self.soft_prompt_len > 0:
                prompt_embeddings_flat = self.soft_prompt(_pair_prompt_ids)
                prompt_embeddings = prompt_embeddings_flat.view(-1, self.soft_prompt_len, self.hidden_size)
                embeddings = torch.cat([input_embeddings, prompt_embeddings], dim=1)
            else:
                embeddings = input_embeddings

            outputs = self.bert_model(inputs_embeds  = embeddings, 
                                      attention_mask = _attention_mask)
            
            #hidden_states = outputs.hidden_states
            sequence_output = outputs[0]
            #pooled_output = outputs[1]

            pooled_relation_token_output  = self.rel_token_poolor(sequence_output, _relation_token_index)
            pooled_direction_token_output = self.dir_token_poolor(sequence_output, _direction_token_index) if self.dir_token_poolor else None
            pooled_novelty_token_output   = self.nov_token_poolor(sequence_output, _novelty_token_index) if self.nov_token_poolor else None

            # use max
            max_pooled_rel_token_output, _ = torch.max(pooled_relation_token_output, dim=0)
            max_pooled_dir_token_output, _ = torch.max(pooled_direction_token_output, dim=0) if pooled_direction_token_output is not None else (None, None)
            max_pooled_nov_token_output, _ = torch.max(pooled_novelty_token_output, dim=0) if pooled_novelty_token_output is not None else (None, None)

            # use attention   
            #max_pooled_rel_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_relation_token_output, pooled_relation_token_output.transpose(-1, -2)), dim=-1), pooled_relation_token_output)[0]
            #max_pooled_dir_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_direction_token_output, pooled_direction_token_output.transpose(-1, -2)), dim=-1), pooled_direction_token_output)[0] if pooled_direction_token_output is not None else None
            #max_pooled_nov_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_novelty_token_output, pooled_novelty_token_output.transpose(-1, -2)), dim=-1), pooled_novelty_token_output)[0] if pooled_novelty_token_output is not None else None

            _rel_token_logits  = self.rel_token_classifier(max_pooled_rel_token_output)
            _dir_token_logits  = self.dir_token_classifier(max_pooled_dir_token_output) if pooled_direction_token_output is not None else None
            _nov_token_logits  = self.nov_token_classifier(max_pooled_nov_token_output) if pooled_novelty_token_output is not None else None

            rel_token_logits.append(_rel_token_logits)
            nov_token_logits.append(_nov_token_logits)
            dir_token_logits.append(_dir_token_logits)
        
        rel_token_logits = torch.stack(rel_token_logits)
        nov_token_logits = torch.stack(nov_token_logits)
        dir_token_logits = torch.stack(dir_token_logits)
        return rel_token_logits, nov_token_logits, dir_token_logits
    
    def __multi_chunk_forward(self, 
                              input_ids, 
                              attention_mask        = None, 
                              pair_prompt_ids       = None,
                              relation_token_index  = None,
                              direction_token_index = None,
                              novelty_token_index   = None):

        batch_size, num_chunks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)            # (B*3, T)
        pair_prompt_ids = pair_prompt_ids.view(-1)         # (B*3,)

        # === Soft prompt embedding concat ===
        input_embeddings = self.bert_model.embeddings.word_embeddings(input_ids)  # (B*3, T, H)

        if self.soft_prompt_len > 0:
            prompt_embeddings_flat = self.soft_prompt(pair_prompt_ids)
            prompt_embeddings = prompt_embeddings_flat.view(-1, self.soft_prompt_len, self.hidden_size)
            embeddings = torch.cat([input_embeddings, prompt_embeddings], dim=1)
        else:
            embeddings = input_embeddings

        attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # (B*3, T)

        outputs = self.bert_model(inputs_embeds  = embeddings, 
                                  attention_mask = attention_mask)
        
        sequence_output = outputs[0]
        sequence_output = sequence_output.view(batch_size, num_chunks, -1, self.hidden_size)

        # === Special token index reshape ===
        relation_token_index  = relation_token_index.view(-1)
        direction_token_index = direction_token_index.view(-1) if direction_token_index is not None else None
        novelty_token_index   = novelty_token_index.view(-1)   if novelty_token_index is not None else None

        pooled_relation_token_output  = self.rel_token_poolor(sequence_output, relation_token_index)
        pooled_direction_token_output = self.dir_token_poolor(sequence_output, direction_token_index) if self.dir_token_poolor and direction_token_index is not None else None
        pooled_novelty_token_output   = self.nov_token_poolor(sequence_output, novelty_token_index)   if self.nov_token_poolor and novelty_token_index is not None else None

        pooled_relation_token_output  = pooled_relation_token_output.view(batch_size, num_chunks, -1)  # (B, 3, H)
        pooled_direction_token_output = pooled_direction_token_output.view(batch_size, num_chunks, -1) if pooled_direction_token_output is not None else None
        pooled_novelty_token_output   = pooled_novelty_token_output.view(batch_size, num_chunks, -1) if pooled_novelty_token_output is not None else None
        
        max_pooled_rel_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_relation_token_output, pooled_relation_token_output.transpose(-1, -2)), dim=-1), pooled_relation_token_output)[:, 0, :]
        max_pooled_dir_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_direction_token_output, pooled_direction_token_output.transpose(-1, -2)), dim=-1), pooled_direction_token_output)[:, 0, :] if pooled_direction_token_output is not None else None
        max_pooled_nov_token_output = torch.matmul(torch.softmax(torch.matmul(pooled_novelty_token_output, pooled_novelty_token_output.transpose(-1, -2)), dim=-1), pooled_novelty_token_output)[:, 0, :] if pooled_novelty_token_output is not None else None

        rel_logits = self.rel_token_classifier(max_pooled_rel_token_output) 
        dir_logits = self.dir_token_classifier(max_pooled_dir_token_output) if max_pooled_dir_token_output is not None else None
        nov_logits = self.nov_token_classifier(max_pooled_nov_token_output) if max_pooled_nov_token_output is not None else None

        return rel_logits, nov_logits, dir_logits
    
    def resize_token_embeddings(self, new_num_tokens):
        self.bert_model.resize_token_embeddings(new_num_tokens)

    def save_model(self, save_path, additional_info=None):
        """Saves the BioREDirect model, its state, and additional information."""
        
        # Create the save directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        self.bert_model.save_pretrained(save_dir)

        # Gather model-specific information (state_dict)
        save_data = {
            #'bert_state_dict': bert_state_dict,
            #'bioredirect_state_dict': custom_state_dict,
            'state_dict':  self.state_dict(),
            'label_to_id': self.label_to_id,
            'relation_label_to_id':  self.relation_label_to_id,
            'novelty_label_to_id':   self.novelty_label_to_id,
            'direction_label_to_id': self.direction_label_to_id,
            'soft_prompt_len': self.soft_prompt_len,
            'num_soft_prompt': self.num_soft_prompt,
            'hidden_size': self.hidden_size,
            'use_single_chunk': self.use_single_chunk
        }

        # Add any extra info you might need for later
        if additional_info:
            save_data.update(additional_info)  
        
        # Save everything to a file
        
        torch.save(save_data, save_path)
        #self.bert_model.config.to_json_file(save_dir + '/config.json')
        print(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, model_path, use_single_chunk=False):
        """Loads the BioREDirect model and its state from a saved file."""

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved data
        saved_data = torch.load(model_path + '/out_bioredirect_model.pth', map_location = device, weights_only = True)
        
        # Determine if a GPU is available, default to 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {device}")
        
        # Initialize a new BioREDirect model instance
        model = BioREDirect(
            in_bert_model         = model_path,
            soft_prompt_len       = saved_data['soft_prompt_len'],
            num_soft_prompt       = saved_data['num_soft_prompt'],
            label_to_id           = saved_data['label_to_id'],
            relation_label_to_id  = saved_data['relation_label_to_id'],
            novelty_label_to_id   = saved_data['novelty_label_to_id'],
            direction_label_to_id = saved_data['direction_label_to_id'],
            hidden_size           = saved_data['hidden_size'],
            use_single_chunk      = saved_data['use_single_chunk'] if 'use_single_chunk' in saved_data else use_single_chunk
        ).to(device)

        model.load_state_dict(saved_data['state_dict'])        
        
        model.to(device)
        
        model.eval()
        return model