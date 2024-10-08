from collections import deque

import networkx as nx 

import torch
import torch.nn.functional as F
import llama_cpp

model = llama_cpp.Llama(
      model_path="./models/Phi-3.5-mini-instruct.Q8_0.gguf",
      n_gpu_layers=-1,
      verbose=True,
      logits_all=True,
)

def get_next_k_tokens(sentence, k=5):
    
    """ Given a sentence get the next top k tokens """

    if isinstance(sentence, str):
        sentence = sentence.encode("utf-8")
    
    model.reset()
    model.eval(model.tokenize(sentence))

    next_logits = torch.tensor(model.eval_logits)[-1,:]

    # approach with softmax to get probabilities
    # next_probs = F.softmax(next_logits, dim=0)
    # top_probs, top_indices = torch.topk(next_probs, k)

    # approach using directly the logits 
    top_values, top_indices = torch.topk(next_logits, k)
    
    next_tokens = [
        model.detokenize([idx]).decode("utf-8") for idx in top_indices
    ]

    return next_tokens


def phrase_tree_search(seed, k, max_depth=3):

    """ Breadth first tree search algorithm to explore phrases """
    
    queue = deque([(seed, 0)])

    while queue:

        current_phrase, depth = queue.popleft()

        if depth > max_depth:
            break

        next_tokens = get_next_k_tokens(current_phrase, k)
        next_phrases = [current_phrase + n for n in next_tokens]

        for phrase in next_phrases:
            queue.append((phrase, depth+1))
    
    return queue


def phrase_graph(seed, k, max_depth=3):

    """ Breadth first tree search algorithm to generate a graph
        that represents the paths """
    
    G = nx.DiGraph()
    G.add_node((seed, 0), phrase=seed, token='<SEED>', depth=0)

    queue = deque([(seed, seed, 0)])

    while queue:
        
        current_p, current_t, depth = queue.popleft()

        if depth >= max_depth:
            break

        next_tokens = get_next_k_tokens(current_p, k)
        next_phrases = [current_p + n for n in next_tokens]
        
        for phrase, token in zip(next_phrases, next_tokens):
            new_depth = depth + 1
            node_id = (token, new_depth)
            G.add_node(node_id, phrase=phrase, token=token, depth=new_depth)
            G.add_edge((current_t, depth), node_id)
            queue.append((phrase, token, new_depth))
    
    return G