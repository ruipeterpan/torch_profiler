import argparse
import torch
import torchvision.models as models

from torchinfo import summary  # torchinfo
from deepspeed.profiling.flops_profiler import get_model_profile  # deepspeed flops profiler
from profiler import TIDSProfiler  # our own profiler


def profile(args):
    with torch.cuda.device(0):
        model = models.resnet18()
        batch_size = 16
        input_size = (batch_size, 3, 224, 224)

        if args.profiler == "torchinfo":
            summary(model, input_size=input_size)
        elif args.profiler == "deepspeed":
            flops, macs, params = get_model_profile(model=model, # model
                                input_shape=input_size, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                args=None, # list of positional arguments to the model.
                                kwargs=None, # dictionary of keyword arguments to the model.
                                print_profile=True, # prints the model graph with the measured profile attached to each module
                                detailed=True, # print the detailed profile
                                module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                top_modules=1, # the number of top modules to print aggregated profile
                                warm_up=10, # the number of warm-ups before measuring the time of each module
                                as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                ignore_modules=None) # the list of modules to ignore in the profiling
        elif args.profiler == "tids":
            inputs = torch.randn(input_size)
            prof = TIDSProfiler(model)
            prof.start_profile()
            model(inputs)
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