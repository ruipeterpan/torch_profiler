import argparse
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from torchinfo import summary  # torchinfo
from deepspeed.profiling.flops_profiler import get_model_profile  # deepspeed flops profiler
from profiler import TIDSProfiler  # our own profiler


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    # inputs: dict with keys "input_ids", "token_type_ids", "attention_mask", "labels"
    return inputs


def profile(args):
    with torch.cuda.device(0):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        batch_size = 1
        seq_len = 128
        if args.profiler == "torchinfo":
            # copied from https://stackoverflow.com/a/68577755/9601555
            summary(model, input_size=(batch_size, seq_len), dtypes=['torch.cuda.IntTensor'])
        elif args.profiler == "deepspeed":
            inputs = bert_input_constructor(batch_size, seq_len, tokenizer)
            flops, macs, params = get_model_profile(
                model,
                kwargs=inputs,
                print_profile=True,
                detailed=True,
                module_depth=-1,
                warm_up=10
            )
        elif args.profiler == "tids":
            inputs = bert_input_constructor(batch_size, seq_len, tokenizer)
            prof = TIDSProfiler(model)
            prof.start_profile()
            model(**inputs)
            profile = prof.generate_profile()
            print(profile)
            prof.end_profile()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profiler",
        type=str,
        default="tids",
        choices=["tids", "torchinfo", "deepspeed"]
    )

    args = parser.parse_args()
    profile(args)


if __name__ == "__main__":
    main()