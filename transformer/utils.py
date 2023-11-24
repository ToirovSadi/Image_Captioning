import torch
from queue import PriorityQueue
from datetime import datetime
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)

def remove_specials(x):
    specials = ['<unk>', '<sos>', '<eos>', '<pad>']
    for sp in specials:
        while sp in x:
            x.remove(sp)
    return x

def generate_ckpt_name(version=None):
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if version is not None:
        name = name + "_" + version
    
    return name

def post_proress(caption, detokenize=True):
    # remove specials
    caption = remove_specials(caption)
    
    # detokenize (join the tokens)
    return TreebankWordDetokenizer().detokenize(caption)

### Beam Search Implementation
class BeamSearchNode:
    def __init__(self, word_id, log_prob, length):
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length
        
    def eval(self, alpha=1):
        return self.log_prob / (self.length ** alpha)
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob

def beam_search(
    src,
    model,
    vocab=None,
    beam_width=5,
    num_candidates=1,
    max_steps=200,
    max_candidates_coef=3,
):
    if vocab is None:
        if hasattr(model, "vocab"):
            vocab = model.vocab
        elif hasattr(model.decoder, "vocab"):
            vocab = model.decoder.vocab
        else:
            raise ValueError("vocab not specified")
    
    qsize = 1
    model.eval()
    candidates = []
    q = PriorityQueue()
    device = model.device
    max_size = model.max_sent_size
    max_candidates = num_candidates * max_candidates_coef
    
    # get encoder outputs
    with torch.no_grad():
        enc_output = model.encoder(src)
    
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    decoder_input = torch.LongTensor([[sos_idx]]).to(device)
    
    # create first node
    node = BeamSearchNode(decoder_input, 0, 1)
    q.put((-node.eval(), node))
    
    with torch.no_grad():
        while not q.empty():
            if qsize > max_steps:
                break
                
            score, node = q.get()
            
            if node.word_id[:, -1] == eos_idx and node.length > 1:
                candidates.append((score, node))
                if len(candidates) >= max_candidates:
                    break
                continue
            if node.length > max_size:
                continue
            
            preds = model.decoder(node.word_id, enc_output)
            preds = torch.log_softmax(preds[:, -1, :], dim=1)
            # preds.shape: [1, output_dim]
            topk, indices = torch.topk(preds, beam_width)
            # topk.shape: [1, beam_width]
            
            # add these topk to the queue
            for i in range(beam_width):
                next_decoder_in = indices[0][i].view(1, -1)
                next_decoder_in = torch.cat((node.word_id, next_decoder_in), dim=1)
                next_prob = topk[0][i].item()
                next_node = BeamSearchNode(
                    next_decoder_in,
                    node.log_prob + next_prob,
                    node.length + 1
                )
                q.put((-next_node.eval(), next_node))
            qsize += beam_width - 1
    
    while len(candidates) < max_candidates and not q.empty():
        candidates.append(q.get())
    
    res = []
    for score, node in candidates:
        ans = node.word_id.cpu().detach().numpy()[0]
        
        ans = vocab.lookup_tokens(ans)
        ans = remove_specials(ans)
        res.append((score, ans))
    
    # sort and take first num_candidates
    candidates = sorted(res, key=lambda x: x[0])[:num_candidates]
    
    # remove scores
    candidates = [token for _, token in candidates]
    return candidates