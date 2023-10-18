import torch
from tqdm import tqdm
import pdb
from torch import nn
import datasets
import transformers
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from arch import Transformer


def preprocess(tok, examples, max_length=100, padding="max_length"):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    model_inputs = tok(inputs, max_length=max_length, truncation=True, padding=padding)
    labels = tok(targets, max_length=max_length, truncation=True, padding=padding)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class LabelSmoothLoss(nn.Module):
    def __init__(self, pad_idx: int = 1, smooth: float = 0.1):
        super().__init__()
        self.smooth = smooth
        self.confidence = 1 - smooth
        self.pad_idx = pad_idx
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_pred, y_true):
        # Smooth
        y_true *= self.confidence
        y_true[y_true == 0] = self.smooth / (y_true.size(-1) - 1)

        # Mask pad tokens
        mask = y_true.argmax(-1) == self.pad_idx
        y_pred = y_pred.clone()[mask]
        y_true = y_true.clone()[mask]
        return self.criterion(nn.functional.log_softmax(y_pred), y_true.to(torch.float))


def train():
    tok = transformers.PreTrainedTokenizerFast.from_pretrained("bpe_tok")
    batch_size = 4
    max_seq_size = 1000
    padding = "longest"
    pad_idx = tok.vocab[tok.pad_token]
    model_params = {
        "src_vocab_size": len(tok.vocab),
        "tgt_vocab_size": len(tok.vocab),
        "embed_size": 512,
        "num_layers": 6,
        "heads": 8,
        "device": "cuda",
        "max_seq_len": max_seq_size,
        "dropout": 0.1,
        "src_pad_idx": pad_idx,
        "tgt_pad_idx": pad_idx,
        "ff_dim": 2048,
    }
    lr = 1e-3
    betas = (0.9, 0.98)
    train_steps = 100_000
    eval_steps = 1_000
    writer = SummaryWriter()

    data = datasets.load_dataset("wmt14", "de-en").with_format("torch")
    train_dataset = data["train"]
    prep_train_dataset = train_dataset.map(
        lambda examples: preprocess(
            tok, examples, max_length=max_seq_size, padding=padding
        ),
        batched=True,
    )
    prep_train_dataset = prep_train_dataset.remove_columns(
        ["translation", "attention_mask", "token_type_ids"]
    )
    val_dataset = data["validation"]
    prep_val_dataset = val_dataset.map(
        lambda examples: preprocess(
            tok, examples, max_length=max_seq_size, padding=padding
        ),
        batched=True,
    )
    prep_val_dataset = prep_val_dataset.remove_columns(
        ["translation", "attention_mask", "token_type_ids"]
    )
    print(prep_train_dataset[0])
    collator = DataCollatorForSeq2Seq(
        tok,
        padding="max_length",
        max_length=max_seq_size,
        label_pad_token_id=tok.vocab[tok.pad_token],
    )
    train_loader = DataLoader(
        prep_train_dataset, batch_size=batch_size, collate_fn=collator
    )
    val_loader = DataLoader(
        prep_val_dataset, batch_size=batch_size, collate_fn=collator
    )

    model = Transformer(**model_params).to(model_params["device"])
    print(model)

    # criterion = nn.KLDivLoss(reduction="batchmean")
    criterion = LabelSmoothLoss(pad_idx=pad_idx, smooth=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-9)

    def rate(step, model_size, factor, warmup):
        """
        we have to default the step to 1 for LambdaLR function
        to avoid zero raising to negative power.
        """
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model_params["embed_size"], factor=1.0, warmup=400
        ),
    )

    steps = 0
    while steps < train_steps:
        for inputs in tqdm(train_loader):
            model.train()
            y_pred = model(inputs["input_ids"], inputs["labels"])
            loss = criterion(
                torch.nn.functional.log_softmax(y_pred.to(torch.float64), dim=-1),
                nn.functional.one_hot(
                    inputs["labels"].to(model_params["device"]),
                    model_params["tgt_vocab_size"],
                ).to(torch.float64),
            )

            writer.add_scalar("train/loss", loss.item(), steps)
            writer.add_scalar("train/seq_len", y_pred.size(1), steps)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            steps += 1

            if steps % eval_steps == 0:
                model.eval()
                losses = []

                with torch.no_grad():
                    for e_inputs in tqdm(val_loader):
                        y_pred = model(e_inputs["input_ids"], e_inputs["labels"])
                        loss = criterion(
                            torch.nn.functional.log_softmax(
                                y_pred.to(torch.float64), dim=-1
                            ),
                            nn.functional.one_hot(
                                e_inputs["labels"].to(model_params["device"]),
                                model_params["tgt_vocab_size"],
                            ).to(torch.float64),
                        )
                        # TODO: calc BLEU and log
                        losses.append(loss.item())
                writer.add_scalar("val/loss", sum(losses) / len(losses), steps)

    writer.flush()


if __name__ == "__main__":
    train()
