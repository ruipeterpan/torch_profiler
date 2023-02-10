# torch_profiler

This profiler combines code from TylerYep/torchinfo ([github](https://github.com/TylerYep/torchinfo)) and Microsoft DeepSpeed's Flops Profiler ([github](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler), [tutorial](https://www.deepspeed.ai/tutorials/flops-profiler/)). The motivation behind writing this up is that DeepSpeed Flops Profiler profiles both the model training/inference speed (latency, throughput) and the efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules but not the shape of the input/output of each module, and torchinfo is the other way around. Although this profiler only provides some basic functionalities, it achieves the best of both worlds in this aspect.

This profiler is based on PyTorch hooks, so the profiling granularity is each `torch.nn.Module`.

## Getting Started/Example

You should first define the PyTorch model and its dummy input:

```python
import torch
import torchvision.models as models
from profiler import TIDSProfiler

# construct model and input
model = models.resnet18()
batch_size = 16
input_size = (batch_size, 3, 224, 224)
inputs = torch.randn(input_size)
```

Then, you can start the profiling. The usage is similar to DeepSpeed Flops Profiler.

```python
# start profiling
prof = TIDSProfiler(model)
prof.start_profile()
model(inputs)
profile = prof.generate_profile()
print(profile)
prof.end_profile()

```

The output in this example looks like:

```text
(model): ResNet, num_params 11689512, 93.666ms, 100.0% latency, input shape [16, 3, 224, 224], output shape [16, 1000]
        (conv1): Conv2d, num_params 9408, 16.923ms, 18.068% latency, input shape [16, 3, 224, 224], output shape [16, 64, 112, 112]
        (bn1): BatchNorm2d, num_params 128, 5.082ms, 5.426% latency, input shape [16, 64, 112, 112], output shape [16, 64, 112, 112]
        (relu): ReLU, num_params 0, 0.86ms, 0.919% latency, input shape [16, 64, 112, 112], output shape [16, 64, 112, 112]
        (maxpool): MaxPool2d, num_params 0, 4.777ms, 5.101% latency, input shape [16, 64, 112, 112], output shape [16, 64, 56, 56]
        (layer1): Sequential, num_params 147968, 20.485ms, 21.87% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                (0): BasicBlock, num_params 73984, 11.09ms, 11.84% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv1): Conv2d, num_params 36864, 4.563ms, 4.871% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn1): BatchNorm2d, num_params 128, 1.42ms, 1.516% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (relu): ReLU, num_params 0, 0.381ms, 0.407% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv2): Conv2d, num_params 36864, 3.379ms, 3.607% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn2): BatchNorm2d, num_params 128, 0.823ms, 0.878% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                (1): BasicBlock, num_params 73984, 9.307ms, 9.937% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv1): Conv2d, num_params 36864, 3.294ms, 3.516% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn1): BatchNorm2d, num_params 128, 0.823ms, 0.878% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (relu): ReLU, num_params 0, 0.381ms, 0.407% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv2): Conv2d, num_params 36864, 3.315ms, 3.539% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn2): BatchNorm2d, num_params 128, 0.979ms, 1.045% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
        (layer2): Sequential, num_params 525568, 15.982ms, 17.062% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                (0): BasicBlock, num_params 230144, 10.653ms, 11.374% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                        (conv1): Conv2d, num_params 73728, 3.164ms, 3.378% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                        (bn1): BatchNorm2d, num_params 256, 0.517ms, 0.552% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (relu): ReLU, num_params 0, 0.193ms, 0.206% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv2): Conv2d, num_params 147456, 2.541ms, 2.713% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn2): BatchNorm2d, num_params 256, 0.48ms, 0.513% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (downsample): Sequential, num_params 8448, 3.474ms, 3.709% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                                (0): Conv2d, num_params 8192, 2.69ms, 2.872% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                                (1): BatchNorm2d, num_params 256, 0.682ms, 0.728% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                (1): BasicBlock, num_params 295424, 5.247ms, 5.602% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv1): Conv2d, num_params 147456, 1.904ms, 2.033% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn1): BatchNorm2d, num_params 256, 0.462ms, 0.493% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (relu): ReLU, num_params 0, 0.196ms, 0.209% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv2): Conv2d, num_params 147456, 2.003ms, 2.138% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn2): BatchNorm2d, num_params 256, 0.437ms, 0.467% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
        (layer3): Sequential, num_params 2099712, 13.916ms, 14.857% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                (0): BasicBlock, num_params 919040, 8.562ms, 9.141% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                        (conv1): Conv2d, num_params 294912, 2.32ms, 2.477% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                        (bn1): BatchNorm2d, num_params 512, 0.353ms, 0.377% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (relu): ReLU, num_params 0, 0.186ms, 0.199% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv2): Conv2d, num_params 589824, 2.581ms, 2.756% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn2): BatchNorm2d, num_params 512, 0.36ms, 0.385% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (downsample): Sequential, num_params 33280, 2.439ms, 2.604% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                                (0): Conv2d, num_params 32768, 1.894ms, 2.022% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                                (1): BatchNorm2d, num_params 512, 0.462ms, 0.493% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                (1): BasicBlock, num_params 1180672, 5.269ms, 5.625% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv1): Conv2d, num_params 589824, 1.952ms, 2.084% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn1): BatchNorm2d, num_params 512, 0.339ms, 0.362% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (relu): ReLU, num_params 0, 0.211ms, 0.226% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv2): Conv2d, num_params 589824, 2.101ms, 2.243% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn2): BatchNorm2d, num_params 512, 0.386ms, 0.412% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
        (layer4): Sequential, num_params 8393728, 13.961ms, 14.905% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                (0): BasicBlock, num_params 3673088, 7.735ms, 8.258% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                        (conv1): Conv2d, num_params 1179648, 2.382ms, 2.543% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                        (bn1): BatchNorm2d, num_params 1024, 0.304ms, 0.325% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (relu): ReLU, num_params 0, 0.201ms, 0.215% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv2): Conv2d, num_params 2359296, 2.95ms, 3.149% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn2): BatchNorm2d, num_params 1024, 0.258ms, 0.275% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (downsample): Sequential, num_params 132096, 1.397ms, 1.492% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                                (0): Conv2d, num_params 131072, 1.038ms, 1.109% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                                (1): BatchNorm2d, num_params 1024, 0.27ms, 0.288% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                (1): BasicBlock, num_params 4720640, 6.148ms, 6.564% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv1): Conv2d, num_params 2359296, 2.594ms, 2.77% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn1): BatchNorm2d, num_params 1024, 0.279ms, 0.298% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (relu): ReLU, num_params 0, 0.166ms, 0.177% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv2): Conv2d, num_params 2359296, 2.604ms, 2.78% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn2): BatchNorm2d, num_params 1024, 0.273ms, 0.291% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
        (avgpool): AdaptiveAvgPool2d, num_params 0, 0.516ms, 0.551% latency, input shape [16, 512, 7, 7], output shape [16, 512, 1, 1]
        (fc): Linear, num_params 513000, 0.663ms, 0.708% latency, input shape [16, 512], output shape [16, 1000]
```

### Comparisons with TorchInfo and DeepSpeed

<details>
  <summary>TorchInfo output</summary>

  ```bash
        $ python3 example_resnet18.py --profiler torchinfo
        ==========================================================================================
        Layer (type:depth-idx)                   Output Shape              Param #
        ==========================================================================================
        ResNet                                   [16, 1000]                --
        ├─Conv2d: 1-1                            [16, 64, 112, 112]        9,408
        ├─BatchNorm2d: 1-2                       [16, 64, 112, 112]        128
        ├─ReLU: 1-3                              [16, 64, 112, 112]        --
        ├─MaxPool2d: 1-4                         [16, 64, 56, 56]          --
        ├─Sequential: 1-5                        [16, 64, 56, 56]          --
        │    └─BasicBlock: 2-1                   [16, 64, 56, 56]          --
        │    │    └─Conv2d: 3-1                  [16, 64, 56, 56]          36,864
        │    │    └─BatchNorm2d: 3-2             [16, 64, 56, 56]          128
        │    │    └─ReLU: 3-3                    [16, 64, 56, 56]          --
        │    │    └─Conv2d: 3-4                  [16, 64, 56, 56]          36,864
        │    │    └─BatchNorm2d: 3-5             [16, 64, 56, 56]          128
        │    │    └─ReLU: 3-6                    [16, 64, 56, 56]          --
        │    └─BasicBlock: 2-2                   [16, 64, 56, 56]          --
        │    │    └─Conv2d: 3-7                  [16, 64, 56, 56]          36,864
        │    │    └─BatchNorm2d: 3-8             [16, 64, 56, 56]          128
        │    │    └─ReLU: 3-9                    [16, 64, 56, 56]          --
        │    │    └─Conv2d: 3-10                 [16, 64, 56, 56]          36,864
        │    │    └─BatchNorm2d: 3-11            [16, 64, 56, 56]          128
        │    │    └─ReLU: 3-12                   [16, 64, 56, 56]          --
        ├─Sequential: 1-6                        [16, 128, 28, 28]         --
        │    └─BasicBlock: 2-3                   [16, 128, 28, 28]         --
        │    │    └─Conv2d: 3-13                 [16, 128, 28, 28]         73,728
        │    │    └─BatchNorm2d: 3-14            [16, 128, 28, 28]         256
        │    │    └─ReLU: 3-15                   [16, 128, 28, 28]         --
        │    │    └─Conv2d: 3-16                 [16, 128, 28, 28]         147,456
        │    │    └─BatchNorm2d: 3-17            [16, 128, 28, 28]         256
        │    │    └─Sequential: 3-18             [16, 128, 28, 28]         8,448
        │    │    └─ReLU: 3-19                   [16, 128, 28, 28]         --
        │    └─BasicBlock: 2-4                   [16, 128, 28, 28]         --
        │    │    └─Conv2d: 3-20                 [16, 128, 28, 28]         147,456
        │    │    └─BatchNorm2d: 3-21            [16, 128, 28, 28]         256
        │    │    └─ReLU: 3-22                   [16, 128, 28, 28]         --
        │    │    └─Conv2d: 3-23                 [16, 128, 28, 28]         147,456
        │    │    └─BatchNorm2d: 3-24            [16, 128, 28, 28]         256
        │    │    └─ReLU: 3-25                   [16, 128, 28, 28]         --
        ├─Sequential: 1-7                        [16, 256, 14, 14]         --
        │    └─BasicBlock: 2-5                   [16, 256, 14, 14]         --
        │    │    └─Conv2d: 3-26                 [16, 256, 14, 14]         294,912
        │    │    └─BatchNorm2d: 3-27            [16, 256, 14, 14]         512
        │    │    └─ReLU: 3-28                   [16, 256, 14, 14]         --
        │    │    └─Conv2d: 3-29                 [16, 256, 14, 14]         589,824
        │    │    └─BatchNorm2d: 3-30            [16, 256, 14, 14]         512
        │    │    └─Sequential: 3-31             [16, 256, 14, 14]         33,280
        │    │    └─ReLU: 3-32                   [16, 256, 14, 14]         --
        │    └─BasicBlock: 2-6                   [16, 256, 14, 14]         --
        │    │    └─Conv2d: 3-33                 [16, 256, 14, 14]         589,824
        │    │    └─BatchNorm2d: 3-34            [16, 256, 14, 14]         512
        │    │    └─ReLU: 3-35                   [16, 256, 14, 14]         --
        │    │    └─Conv2d: 3-36                 [16, 256, 14, 14]         589,824
        │    │    └─BatchNorm2d: 3-37            [16, 256, 14, 14]         512
        │    │    └─ReLU: 3-38                   [16, 256, 14, 14]         --
        ├─Sequential: 1-8                        [16, 512, 7, 7]           --
        │    └─BasicBlock: 2-7                   [16, 512, 7, 7]           --
        │    │    └─Conv2d: 3-39                 [16, 512, 7, 7]           1,179,648
        │    │    └─BatchNorm2d: 3-40            [16, 512, 7, 7]           1,024
        │    │    └─ReLU: 3-41                   [16, 512, 7, 7]           --
        │    │    └─Conv2d: 3-42                 [16, 512, 7, 7]           2,359,296
        │    │    └─BatchNorm2d: 3-43            [16, 512, 7, 7]           1,024
        │    │    └─Sequential: 3-44             [16, 512, 7, 7]           132,096
        │    │    └─ReLU: 3-45                   [16, 512, 7, 7]           --
        │    └─BasicBlock: 2-8                   [16, 512, 7, 7]           --
        │    │    └─Conv2d: 3-46                 [16, 512, 7, 7]           2,359,296
        │    │    └─BatchNorm2d: 3-47            [16, 512, 7, 7]           1,024
        │    │    └─ReLU: 3-48                   [16, 512, 7, 7]           --
        │    │    └─Conv2d: 3-49                 [16, 512, 7, 7]           2,359,296
        │    │    └─BatchNorm2d: 3-50            [16, 512, 7, 7]           1,024
        │    │    └─ReLU: 3-51                   [16, 512, 7, 7]           --
        ├─AdaptiveAvgPool2d: 1-9                 [16, 512, 1, 1]           --
        ├─Linear: 1-10                           [16, 1000]                513,000
        ==========================================================================================
        Total params: 11,689,512
        Trainable params: 11,689,512
        Non-trainable params: 0
        Total mult-adds (G): 29.03
        ==========================================================================================
        Input size (MB): 9.63
        Forward/backward pass size (MB): 635.96
        Params size (MB): 46.76
        Estimated Total Size (MB): 692.35
        ==========================================================================================
  ```  
</details>

<details>
  <summary>DeepSpeed output</summary>

  ```bash
    $ python3 example_resnet18.py --profiler deepspeed

    -------------------------- DeepSpeed Flops Profiler --------------------------
    Profile Summary at step 10:
    Notations:
    data parallel size (dp_size), model parallel size(mp_size),
    number of parameters (params), number of multiply-accumulate operations(MACs),
    number of floating-point operations (flops), floating-point operations per second (FLOPS),
    fwd latency (forward propagation latency), bwd latency (backward propagation latency),
    step (weights update latency), iter latency (sum of fwd, bwd and step latency)

    params per gpu:                                               11.69 M 
    params of model = params per GPU * mp_size:                   11.69 M 
    fwd MACs per GPU:                                             29.03 GMACs
    fwd flops per GPU:                                            58.18 G 
    fwd flops of model = fwd flops per GPU * mp_size:             58.18 G 
    fwd latency:                                                  80.71 ms
    fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          720.81 GFLOPS

    ----------------------------- Aggregated Profile per GPU -----------------------------
    Top 1 modules in terms of params, MACs or fwd latency at different model depths:
    depth 0:
        params      - {'ResNet': '11.69 M'}
        MACs        - {'ResNet': '29.03 GMACs'}
        fwd latency - {'ResNet': '80.71 ms'}
    depth 1:
        params      - {'Sequential': '11.17 M'}
        MACs        - {'Sequential': '27.13 GMACs'}
        fwd latency - {'Sequential': '63.32 ms'}
    depth 2:
        params      - {'BasicBlock': '11.17 M'}
        MACs        - {'BasicBlock': '27.13 GMACs'}
        fwd latency - {'BasicBlock': '63.07 ms'}
    depth 3:
        params      - {'Conv2d': '10.99 M'}
        MACs        - {'Conv2d': '26.82 GMACs'}
        fwd latency - {'Conv2d': '50.08 ms'}

    ------------------------------ Detailed Profile per GPU ------------------------------
    Each module profile is listed after its name in the following order: 
    params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

    Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
    2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
    3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

    ResNet(
    11.69 M, 100.00% Params, 29.03 GMACs, 100.00% MACs, 80.71 ms, 100.00% latency, 720.81 GFLOPS, 
    (conv1): Conv2d(9.41 k, 0.08% Params, 1.89 GMACs, 6.51% MACs, 9.87 ms, 12.22% latency, 382.76 GFLOPS, 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 3.26 ms, 4.04% latency, 7.88 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 713.11 us, 0.88% latency, 18.01 GFLOPS, inplace=True)
    (maxpool): MaxPool2d(0, 0.00% Params, 0 MACs, 0.00% MACs, 2.6 ms, 3.23% latency, 4.93 GFLOPS, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
        147.97 k, 1.27% Params, 7.4 GMACs, 25.49% MACs, 21.73 ms, 26.92% latency, 682.68 GFLOPS, 
        (0): BasicBlock(
        73.98 k, 0.63% Params, 3.7 GMACs, 12.75% MACs, 12.98 ms, 16.09% latency, 571.3 GFLOPS, 
        (conv1): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 3.15 ms, 3.90% latency, 1.18 TFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 355.24 us, 0.44% latency, 18.08 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 329.49 us, 0.41% latency, 19.49 GFLOPS, inplace=True)
        (conv2): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 8.32 ms, 10.31% latency, 444.68 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 380.28 us, 0.47% latency, 16.89 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
        73.98 k, 0.63% Params, 3.7 GMACs, 12.75% MACs, 8.69 ms, 10.76% latency, 854.04 GFLOPS, 
        (conv1): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 3.77 ms, 4.67% latency, 981.61 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 342.37 us, 0.42% latency, 18.76 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 319.72 us, 0.40% latency, 20.09 GFLOPS, inplace=True)
        (conv2): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 3.48 ms, 4.31% latency, 1.06 TFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 356.67 us, 0.44% latency, 18.01 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )
    (layer2): Sequential(
        525.57 k, 4.50% Params, 6.58 GMACs, 22.66% MACs, 16.06 ms, 19.90% latency, 820.22 GFLOPS, 
        (0): BasicBlock(
        230.14 k, 1.97% Params, 2.88 GMACs, 9.91% MACs, 10.08 ms, 12.49% latency, 572.07 GFLOPS, 
        (conv1): Conv2d(73.73 k, 0.63% Params, 924.84 MMACs, 3.19% MACs, 3.28 ms, 4.06% latency, 563.9 GFLOPS, 64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 236.27 us, 0.29% latency, 13.59 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 183.82 us, 0.23% latency, 17.47 GFLOPS, inplace=True)
        (conv2): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 3.29 ms, 4.08% latency, 1.12 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 131.13 us, 0.16% latency, 24.49 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
            8.45 k, 0.07% Params, 102.76 MMACs, 0.35% MACs, 2.68 ms, 3.32% latency, 77.89 GFLOPS, 
            (0): Conv2d(8.19 k, 0.07% Params, 102.76 MMACs, 0.35% MACs, 2.29 ms, 2.84% latency, 89.76 GFLOPS, 64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 327.35 us, 0.41% latency, 9.81 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        )
        (1): BasicBlock(
        295.42 k, 2.53% Params, 3.7 GMACs, 12.75% MACs, 5.92 ms, 7.34% latency, 1.25 TFLOPS, 
        (conv1): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 2.66 ms, 3.29% latency, 1.39 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 135.18 us, 0.17% latency, 23.75 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 161.65 us, 0.20% latency, 19.87 GFLOPS, inplace=True)
        (conv2): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 2.64 ms, 3.27% latency, 1.4 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 122.55 us, 0.15% latency, 26.2 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )
    (layer3): Sequential(
        2.1 M, 17.96% Params, 6.58 GMACs, 22.66% MACs, 13.48 ms, 16.70% latency, 976.54 GFLOPS, 
        (0): BasicBlock(
        919.04 k, 7.86% Params, 2.88 GMACs, 9.91% MACs, 9.02 ms, 11.17% latency, 638.8 GFLOPS, 
        (conv1): Conv2d(294.91 k, 2.52% Params, 924.84 MMACs, 3.19% MACs, 4.23 ms, 5.24% latency, 437.4 GFLOPS, 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 144.72 us, 0.18% latency, 11.09 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 165.46 us, 0.20% latency, 9.7 GFLOPS, inplace=True)
        (conv2): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 2.08 ms, 2.58% latency, 1.78 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 125.65 us, 0.16% latency, 12.78 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
            33.28 k, 0.28% Params, 102.76 MMACs, 0.35% MACs, 2.0 ms, 2.47% latency, 103.72 GFLOPS, 
            (0): Conv2d(32.77 k, 0.28% Params, 102.76 MMACs, 0.35% MACs, 1.71 ms, 2.12% latency, 120.04 GFLOPS, 128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 213.15 us, 0.26% latency, 7.53 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        )
        (1): BasicBlock(
        1.18 M, 10.10% Params, 3.7 GMACs, 12.75% MACs, 4.4 ms, 5.45% latency, 1.68 TFLOPS, 
        (conv1): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 1.82 ms, 2.26% latency, 2.03 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 121.12 us, 0.15% latency, 13.26 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 158.55 us, 0.20% latency, 10.13 GFLOPS, inplace=True)
        (conv2): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 1.95 ms, 2.42% latency, 1.89 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 113.25 us, 0.14% latency, 14.18 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )
    (layer4): Sequential(
        8.39 M, 71.81% Params, 6.58 GMACs, 22.66% MACs, 12.04 ms, 14.92% latency, 1.09 TFLOPS, 
        (0): BasicBlock(
        3.67 M, 31.42% Params, 2.88 GMACs, 9.91% MACs, 6.61 ms, 8.19% latency, 870.77 GFLOPS, 
        (conv1): Conv2d(1.18 M, 10.09% Params, 924.84 MMACs, 3.19% MACs, 2.04 ms, 2.53% latency, 906.11 GFLOPS, 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 165.7 us, 0.21% latency, 4.84 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 151.63 us, 0.19% latency, 5.29 GFLOPS, inplace=True)
        (conv2): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.55 ms, 3.17% latency, 1.45 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 130.41 us, 0.16% latency, 6.16 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
            132.1 k, 1.13% Params, 102.76 MMACs, 0.35% MACs, 1.32 ms, 1.63% latency, 156.46 GFLOPS, 
            (0): Conv2d(131.07 k, 1.12% Params, 102.76 MMACs, 0.35% MACs, 1.15 ms, 1.42% latency, 179.1 GFLOPS, 256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 107.05 us, 0.13% latency, 7.5 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        )
        (1): BasicBlock(
        4.72 M, 40.38% Params, 3.7 GMACs, 12.75% MACs, 5.37 ms, 6.65% latency, 1.38 TFLOPS, 
        (conv1): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.39 ms, 2.96% latency, 1.55 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 111.34 us, 0.14% latency, 7.21 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 141.86 us, 0.18% latency, 5.66 GFLOPS, inplace=True)
        (conv2): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.42 ms, 3.00% latency, 1.53 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 104.43 us, 0.13% latency, 7.69 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    )
    (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 0 MACs, 0.00% MACs, 210.29 us, 0.26% latency, 1.91 GFLOPS, output_size=(1, 1))
    (fc): Linear(513.0 k, 4.39% Params, 8.19 MMACs, 0.03% MACs, 363.35 us, 0.45% latency, 45.09 GFLOPS, in_features=512, out_features=1000, bias=True)
    )
    ------------------------------------------------------------------------------
  ```
</details>



## Requirements

My development environment is:
* Python 3.9.13
* deepspeed 0.8.0
* torch 1.13.1
* torchinfo 1.7.2
* torchvision 0.14.1
