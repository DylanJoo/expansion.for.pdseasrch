import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

class ContrieverEncoder:
    def __init__(self, model_name='facebook/contriever-msmarco'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(self, sentences: list) -> torch.Tensor:
        # Apply tokenizer
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

        # Compute token embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Generate embeddings using mean pooling
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        
        # Normalize embeddings using L2 norm
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings