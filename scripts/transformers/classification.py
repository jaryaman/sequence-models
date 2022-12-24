from transformers import AutoTokenizer, AutoConfig

from src.seq.transformer.heads import TransformerForSequenceClassification
from src.seq.transformer.model_simplified import Embeddings

if __name__ == '__main__':
    text = "time flies like an arrow"

    model_ckpt = "bert-base-uncased"
    config_bert = AutoConfig.from_pretrained(model_ckpt)
    config_bert.num_labels = 3
    encoder_classifier = TransformerForSequenceClassification(config_bert)

    token_emb = Embeddings(config_bert)  # with positional representations
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    print(encoder_classifier(inputs.input_ids).size())
