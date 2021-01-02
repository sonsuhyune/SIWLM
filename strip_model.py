#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import argparse

def strip_config(parser):
    parser.add_argument('--fout',  help='path for strip model')
    parser.add_argument('--model_path',   help='path to multi-task learned model')
    
    return parser
parser = argparse.ArgumentParser()
parser = strip_config(parser)
args = parser.parse_args()

fout = args.fout
model_path = args.model_path
#fout = 'checkpoints_mnli_matched,mnli_mismatched/mnli_matched,mnli_mismatched_adamax_5e-5_2020-10-08T1731/strip_model.pt'
#model_path = 'checkpoints_mnli_matched,mnli_mismatched/mnli_matched,mnli_mismatched_adamax_5e-5_2020-10-08T1731/model_2.pt'

def main():
    state_dict = torch.load(model_path)

    config = state_dict['config']
    state_dict.keys()

    new_state_dict = {'state': state_dict['state'], 'config': state_dict['config']}
    old_state_dict = {}
    for key, val in new_state_dict['state'].items():
        prefix = key.split('.')[0]
        if prefix == 'scoring_list':
            continue
        old_state_dict[key] = val

    my_config = {}
    my_config['vocab_size'] = config['vocab_size']
    my_config['hidden_size'] = config['hidden_size']
    my_config['num_hidden_layers'] = config['num_hidden_layers']
    my_config['num_attention_heads'] = config['num_attention_heads']
    my_config['hidden_act'] = config['hidden_act']
    my_config['intermediate_size'] = config['intermediate_size']
    my_config['hidden_dropout_prob'] = config['hidden_dropout_prob']
    my_config['attention_probs_dropout_prob'] = config['attention_probs_dropout_prob']
    my_config['max_position_embeddings'] = config['max_position_embeddings']
    my_config['type_vocab_size'] = config['type_vocab_size']
    my_config['initializer_range'] = config['initializer_range']
    state_dict = {'state': old_state_dict, 'config': my_config}
    torch.save(state_dict, fout)

if __name__ == '__main__':
    main()
