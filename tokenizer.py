import tokenizers
import transformers
from tokenizers import models, pre_tokenizers, trainers, decoders, processors
import datasets


if __name__ == "__main__":
    data = datasets.load_dataset("wmt14", "de-en")
    batch_size = 1_000
    vocab_size = 37_000
    unk_token = "[UNK]"
    pad_token = "[PAD]"

    tok = tokenizers.Tokenizer(models.BPE(unk_token=unk_token))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel()
    tok.decoder = decoders.ByteLevel()

    def batch_iterator():
        for i in range(0, len(data), batch_size):
            yield [
                " ".join([x["de"], x["en"]])
                for x in data["train"][i : i + batch_size]["translation"]
            ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[unk_token, pad_token]
    )

    tok.train_from_iterator(batch_iterator(), trainer=trainer)
    tok.enable_padding(pad_id=tok.token_to_id(pad_token), pad_token=pad_token)

    sent = "Succesfully created BPE tokenizer :)"
    enc = tok.encode(sent)
    print(f"Encoded sentence: {enc.tokens}")
    print(f"Decoded sentence: {tok.decode(enc.ids)}")

    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok)
    tok.save_pretrained("bpe_tok")
