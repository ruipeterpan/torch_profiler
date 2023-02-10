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


### BERT example

<details>
  <summary>Profiler output</summary>

  ```bash
        $ python3 example_bert.py
        (model): BertForSequenceClassification, num_params 109483778, 294.45ms, 100.0% latency, input shape [], output shape [2, 2]
        (bert): BertModel, num_params 109482240, 293.658ms, 99.731% latency, input shape [2, 512], output shape [2, 768]
                (embeddings): BertEmbeddings, num_params 23837184, 3.185ms, 1.082% latency, input shape [], output shape [2, 512, 768]
                        (word_embeddings): Embedding, num_params 23440896, 1.011ms, 0.343% latency, input shape [2, 512], output shape [2, 512, 768]
                        (position_embeddings): Embedding, num_params 393216, 0.323ms, 0.11% latency, input shape [1, 512], output shape [1, 512, 768]
                        (token_type_embeddings): Embedding, num_params 1536, 0.352ms, 0.12% latency, input shape [2, 512], output shape [2, 512, 768]
                        (LayerNorm): LayerNorm, num_params 1536, 0.615ms, 0.209% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                        (dropout): Dropout, num_params 0, 0.096ms, 0.033% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                (encoder): BertEncoder, num_params 85054464, 285.609ms, 96.998% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                        (layer): ModuleList, num_params 85054464, 285.113ms, 96.829% latency, input shape None, output shape None
                                (0): BertLayer, num_params 7087872, 24.88ms, 8.45% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 16.394ms, 5.568% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 14.743ms, 5.007% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.6ms, 0.543% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 1.17ms, 0.397% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 1.127ms, 0.383% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.094ms, 0.032% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 1.549ms, 0.526% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.018ms, 0.346% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.24ms, 0.082% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.059ms, 0.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.808ms, 1.293% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 2.674ms, 0.908% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 1.04ms, 0.353% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.461ms, 1.515% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.944ms, 1.339% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.215ms, 0.073% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.051ms, 0.017% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (1): BertLayer, num_params 7087872, 20.387ms, 6.924% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 13.286ms, 4.512% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 11.699ms, 3.973% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 0.943ms, 0.32% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.784ms, 0.266% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.78ms, 0.265% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.096ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 1.496ms, 0.508% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.002ms, 0.34% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.202ms, 0.069% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.064ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.174ms, 1.078% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 2.689ms, 0.913% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.4ms, 0.136% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 3.719ms, 1.263% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.17ms, 1.077% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.265ms, 0.09% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.051ms, 0.017% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (2): BertLayer, num_params 7087872, 25.887ms, 8.792% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 18.315ms, 6.22% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 16.593ms, 5.635% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 0.961ms, 0.326% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.766ms, 0.26% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.76ms, 0.258% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.086ms, 0.029% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 1.631ms, 0.554% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.111ms, 0.377% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.216ms, 0.073% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.061ms, 0.021% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.627ms, 1.232% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.08ms, 1.046% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.459ms, 0.156% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 3.752ms, 1.274% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.14ms, 1.066% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.262ms, 0.089% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.08ms, 0.027% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (3): BertLayer, num_params 7087872, 21.844ms, 7.419% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 13.742ms, 4.667% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 11.742ms, 3.988% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 0.969ms, 0.329% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.798ms, 0.271% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.797ms, 0.271% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.096ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 1.906ms, 0.647% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.294ms, 0.439% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.268ms, 0.091% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.06ms, 0.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.89ms, 1.321% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.264ms, 1.109% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.537ms, 0.182% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.013ms, 1.363% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.418ms, 1.161% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.27ms, 0.092% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.05ms, 0.017% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (4): BertLayer, num_params 7087872, 22.564ms, 7.663% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.133ms, 4.8% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 11.937ms, 4.054% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.013ms, 0.344% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.829ms, 0.282% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.81ms, 0.275% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.091ms, 0.031% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.101ms, 0.713% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.409ms, 0.478% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.272ms, 0.092% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.069ms, 0.023% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.066ms, 1.381% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.417ms, 1.161% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.553ms, 0.188% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.17ms, 1.416% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.526ms, 1.197% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.281ms, 0.095% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.053ms, 0.018% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (5): BertLayer, num_params 7087872, 22.896ms, 7.776% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.434ms, 4.902% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.332ms, 4.188% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.12ms, 0.38% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.811ms, 0.275% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.845ms, 0.287% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.093ms, 0.032% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 1.996ms, 0.678% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.36ms, 0.462% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.257ms, 0.087% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.06ms, 0.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.062ms, 1.38% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.411ms, 1.158% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.561ms, 0.19% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.195ms, 1.425% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.586ms, 1.218% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.251ms, 0.085% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.064ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (6): BertLayer, num_params 7087872, 23.066ms, 7.834% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.28ms, 4.85% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.083ms, 4.104% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.086ms, 0.369% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.818ms, 0.278% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.826ms, 0.281% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.097ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.088ms, 0.709% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.405ms, 0.477% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.254ms, 0.086% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.067ms, 0.023% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.222ms, 1.434% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.506ms, 1.191% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.627ms, 0.213% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.362ms, 1.481% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.671ms, 1.247% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.279ms, 0.095% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.069ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (7): BertLayer, num_params 7087872, 23.795ms, 8.081% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.991ms, 5.091% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.735ms, 4.325% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.212ms, 0.412% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.821ms, 0.279% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.885ms, 0.3% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.097ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.16ms, 0.734% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.487ms, 0.505% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.258ms, 0.088% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.059ms, 0.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.217ms, 1.432% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.509ms, 1.192% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.62ms, 0.211% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.388ms, 1.49% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.71ms, 1.26% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.301ms, 0.102% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.065ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (8): BertLayer, num_params 7087872, 23.535ms, 7.993% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.743ms, 5.007% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.563ms, 4.267% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.196ms, 0.406% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.842ms, 0.286% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.874ms, 0.297% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.096ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.086ms, 0.708% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.422ms, 0.483% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.254ms, 0.086% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.059ms, 0.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.29ms, 1.457% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.575ms, 1.214% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.627ms, 0.213% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.303ms, 1.461% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.634ms, 1.234% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.268ms, 0.091% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.064ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (9): BertLayer, num_params 7087872, 23.326ms, 7.922% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.817ms, 5.032% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.528ms, 4.255% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.127ms, 0.383% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.874ms, 0.297% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.864ms, 0.293% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.097ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.194ms, 0.745% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.527ms, 0.519% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.246ms, 0.083% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.057ms, 0.019% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.14ms, 1.406% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.431ms, 1.165% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.619ms, 0.21% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.165ms, 1.415% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.499ms, 1.188% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.265ms, 0.09% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.062ms, 0.021% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (10): BertLayer, num_params 7087872, 23.515ms, 7.986% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 14.851ms, 5.044% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.582ms, 4.273% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.184ms, 0.402% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.824ms, 0.28% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.879ms, 0.299% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.096ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.168ms, 0.736% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.481ms, 0.503% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.268ms, 0.091% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.072ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.174ms, 1.418% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.484ms, 1.183% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.6ms, 0.204% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.294ms, 1.458% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.618ms, 1.229% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.281ms, 0.095% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.064ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (11): BertLayer, num_params 7087872, 29.417ms, 9.991% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 20.554ms, 6.98% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.602ms, 4.28% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.112ms, 0.378% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.853ms, 0.29% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.904ms, 0.307% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.097ms, 0.033% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 7.848ms, 2.665% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.451ms, 0.493% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 5.954ms, 2.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.072ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.271ms, 1.45% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.522ms, 1.196% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.645ms, 0.219% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.353ms, 1.478% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.632ms, 1.233% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.292ms, 0.099% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.102ms, 0.035% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                (pooler): BertPooler, num_params 590592, 4.198ms, 1.426% latency, input shape [2, 512, 768], output shape [2, 768]
                        (dense): Linear, num_params 590592, 3.887ms, 1.32% latency, input shape [2, 768], output shape [2, 768]
                        (activation): Tanh, num_params 0, 0.184ms, 0.062% latency, input shape [2, 768], output shape [2, 768]
        (dropout): Dropout, num_params 0, 0.052ms, 0.018% latency, input shape [2, 768], output shape [2, 768]
        (classifier): Linear, num_params 1538, 0.086ms, 0.029% latency, input shape [2, 768], output shape [2, 2]
  ```  
</details>

<details>
  <summary>TorchInfo output</summary>

  ```bash
        $ python3 example_bert.py --profiler torchinfo
        =========================================================================================================
        Layer (type:depth-idx)                                  Output Shape              Param #
        =========================================================================================================
        BertForSequenceClassification                           [2, 2]                    --
        ├─BertModel: 1-1                                        [2, 768]                  --
        │    └─BertEmbeddings: 2-1                              [2, 512, 768]             --
        │    │    └─Embedding: 3-1                              [2, 512, 768]             23,440,896
        │    │    └─Embedding: 3-2                              [2, 512, 768]             1,536
        │    │    └─Embedding: 3-3                              [1, 512, 768]             393,216
        │    │    └─LayerNorm: 3-4                              [2, 512, 768]             1,536
        │    │    └─Dropout: 3-5                                [2, 512, 768]             --
        │    └─BertEncoder: 2-2                                 [2, 512, 768]             --
        │    │    └─ModuleList: 3-6                             --                        85,054,464
        │    └─BertPooler: 2-3                                  [2, 768]                  --
        │    │    └─Linear: 3-7                                 [2, 768]                  590,592
        │    │    └─Tanh: 3-8                                   [2, 768]                  --
        ├─Dropout: 1-2                                          [2, 768]                  --
        ├─Linear: 1-3                                           [2, 2]                    1,538
        =========================================================================================================
        Total params: 109,483,778
        Trainable params: 109,483,778
        Non-trainable params: 0
        Total mult-adds (M): 218.57
        =========================================================================================================
        Input size (MB): 0.00
        Forward/backward pass size (MB): 852.50
        Params size (MB): 437.94
        Estimated Total Size (MB): 1290.44
        =========================================================================================================
  ```  
</details>


<details>
  <summary>DeepSpeed output</summary>

  ```bash
        $ python3 example_bert.py --profiler deepspeed
        
        -------------------------- DeepSpeed Flops Profiler --------------------------
        Profile Summary at step 10:
        Notations:
        data parallel size (dp_size), model parallel size(mp_size),
        number of parameters (params), number of multiply-accumulate operations(MACs),
        number of floating-point operations (flops), floating-point operations per second (FLOPS),
        fwd latency (forward propagation latency), bwd latency (backward propagation latency),
        step (weights update latency), iter latency (sum of fwd, bwd and step latency)

        params per gpu:                                               109.48 M
        params of model = params per GPU * mp_size:                   109.48 M
        fwd MACs per GPU:                                             96.64 GMACs
        fwd flops per GPU:                                            193.45 G
        fwd flops of model = fwd flops per GPU * mp_size:             193.45 G
        fwd latency:                                                  478.29 ms
        fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          404.46 GFLOPS

        ----------------------------- Aggregated Profile per GPU -----------------------------
        Top 1 modules in terms of params, MACs or fwd latency at different model depths:
        depth 0:
            params      - {'BertForSequenceClassification': '109.48 M'}
            MACs        - {'BertForSequenceClassification': '96.64 GMACs'}
            fwd latency - {'BertForSequenceClassification': '478.29 ms'}
        depth 1:
            params      - {'BertModel': '109.48 M'}
            MACs        - {'BertModel': '96.64 GMACs'}
            fwd latency - {'BertModel': '477.61 ms'}
        depth 2:
            params      - {'BertEncoder': '85.05 M'}
            MACs        - {'BertEncoder': '96.64 GMACs'}
            fwd latency - {'BertEncoder': '472.84 ms'}
        depth 3:
            params      - {'ModuleList': '85.05 M'}
            MACs        - {'ModuleList': '96.64 GMACs'}
            fwd latency - {'ModuleList': '472.28 ms'}
        depth 4:
            params      - {'BertLayer': '85.05 M'}
            MACs        - {'BertLayer': '96.64 GMACs'}
            fwd latency - {'BertLayer': '472.28 ms'}
        depth 5:
            params      - {'BertAttention': '28.37 M'}
            MACs        - {'BertAttention': '38.65 GMACs'}
            fwd latency - {'BertOutput': '210.95 ms'}
        depth 6:
            params      - {'Linear': '56.67 M'}
            MACs        - {'Linear': '57.98 GMACs'}
            fwd latency - {'Linear': '255.46 ms'}

        ------------------------------ Detailed Profile per GPU ------------------------------
        Each module profile is listed after its name in the following order: 
        params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

        Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
        2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
        3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

        BertForSequenceClassification(
          109.48 M, 100.00% Params, 96.64 GMACs, 100.00% MACs, 478.29 ms, 100.00% latency, 404.46 GFLOPS, 
          (bert): BertModel(
            109.48 M, 100.00% Params, 96.64 GMACs, 100.00% MACs, 477.61 ms, 99.86% latency, 405.04 GFLOPS, 
            (embeddings): BertEmbeddings(
              23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.34 ms, 0.28% latency, 2.93 GFLOPS, 
              (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 410.8 us, 0.09% latency, 0.0 FLOPS, 30522, 768, padding_idx=0)
              (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 110.63 us, 0.02% latency, 0.0 FLOPS, 512, 768)
              (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 128.03 us, 0.03% latency, 0.0 FLOPS, 2, 768)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 243.19 us, 0.05% latency, 16.17 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 40.05 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (encoder): BertEncoder(
              85.05 M, 77.69% Params, 96.64 GMACs, 100.00% MACs, 472.84 ms, 98.86% latency, 409.11 GFLOPS, 
              (layer): ModuleList(
                85.05 M, 77.69% Params, 96.64 GMACs, 100.00% MACs, 472.28 ms, 98.75% latency, 409.59 GFLOPS, 
                (0): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 15.57 ms, 3.25% latency, 1.04 TFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 9.11 ms, 1.90% latency, 708.61 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 7.69 ms, 1.61% latency, 681.32 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 857.59 us, 0.18% latency, 1.41 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 649.69 us, 0.14% latency, 1.86 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 697.61 us, 0.15% latency, 1.73 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 53.88 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.34 ms, 0.28% latency, 906.55 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 872.14 us, 0.18% latency, 1.39 TFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 176.19 us, 0.04% latency, 22.32 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.72 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.75 ms, 0.57% latency, 1.76 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.33 ms, 0.49% latency, 2.08 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 361.92 us, 0.08% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.53 ms, 0.74% latency, 1.37 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.09 ms, 0.65% latency, 1.57 TFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 198.84 us, 0.04% latency, 19.78 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 31.23 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (1): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 18.19 ms, 3.80% latency, 886.44 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 11.77 ms, 2.46% latency, 548.41 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 10.31 ms, 2.16% latency, 508.11 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 824.69 us, 0.17% latency, 1.46 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 678.54 us, 0.14% latency, 1.78 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 659.94 us, 0.14% latency, 1.83 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 55.55 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.37 ms, 0.29% latency, 887.09 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 883.82 us, 0.18% latency, 1.37 TFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 203.13 us, 0.04% latency, 19.36 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.95 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.74 ms, 0.57% latency, 1.76 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.28 ms, 0.48% latency, 2.11 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 399.11 us, 0.08% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.51 ms, 0.73% latency, 1.38 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.08 ms, 0.64% latency, 1.57 TFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 189.78 us, 0.04% latency, 20.72 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 31.95 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (2): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.79 ms, 4.77% latency, 707.25 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 16.47 ms, 3.44% latency, 391.75 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 15.01 ms, 3.14% latency, 349.06 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 844.48 us, 0.18% latency, 1.43 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 691.18 us, 0.14% latency, 1.75 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 669.0 us, 0.14% latency, 1.81 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 52.21 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.39 ms, 0.29% latency, 872.78 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 910.04 us, 0.19% latency, 1.33 TFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 197.41 us, 0.04% latency, 19.92 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.72 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.7 ms, 0.57% latency, 1.79 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.27 ms, 0.48% latency, 2.12 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 368.6 us, 0.08% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.44 ms, 0.72% latency, 1.41 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.02 ms, 0.63% latency, 1.6 TFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 173.33 us, 0.04% latency, 22.69 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 30.99 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (3): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 43.38 ms, 9.07% latency, 371.59 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.95 ms, 3.13% latency, 431.48 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.51 ms, 2.83% latency, 387.82 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 849.01 us, 0.18% latency, 1.42 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 680.45 us, 0.14% latency, 1.78 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.99 ms, 0.83% latency, 302.9 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 55.31 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.36 ms, 0.28% latency, 893.02 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 864.27 us, 0.18% latency, 1.4 TFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 202.42 us, 0.04% latency, 19.43 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.34 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.19 ms, 1.09% latency, 930.97 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.74 ms, 0.99% latency, 1.02 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 394.11 us, 0.08% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.07 ms, 4.82% latency, 209.59 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.58 ms, 4.72% latency, 213.96 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 196.46 us, 0.04% latency, 20.02 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 40.29 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (4): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 40.8 ms, 8.53% latency, 395.06 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 12.35 ms, 2.58% latency, 522.64 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 10.7 ms, 2.24% latency, 489.7 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 815.39 us, 0.17% latency, 1.48 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 683.07 us, 0.14% latency, 1.77 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 661.37 us, 0.14% latency, 1.83 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 55.31 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.57 ms, 0.33% latency, 770.62 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.09 ms, 0.23% latency, 1.11 TFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 194.79 us, 0.04% latency, 20.19 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.15 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.49 ms, 1.15% latency, 880.18 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.88 ms, 1.02% latency, 991.06 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 552.89 us, 0.12% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.8 ms, 4.77% latency, 212.14 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.24 ms, 4.65% latency, 217.21 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 242.23 us, 0.05% latency, 16.23 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 35.29 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (5): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 42.71 ms, 8.93% latency, 377.42 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 13.21 ms, 2.76% latency, 488.29 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 11.26 ms, 2.35% latency, 465.4 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.0 ms, 0.21% latency, 1.2 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 733.85 us, 0.15% latency, 1.65 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 714.06 us, 0.15% latency, 1.69 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.74 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.88 ms, 0.39% latency, 645.96 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.28 ms, 0.27% latency, 945.25 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 254.87 us, 0.05% latency, 15.43 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.19 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 6.07 ms, 1.27% latency, 796.35 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.46 ms, 1.14% latency, 885.03 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 541.21 us, 0.11% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.24 ms, 4.86% latency, 208.09 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.73 ms, 4.75% latency, 212.58 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 212.91 us, 0.04% latency, 18.47 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 32.42 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (6): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 33.23 ms, 6.95% latency, 485.09 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 17.6 ms, 3.68% latency, 366.55 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 15.25 ms, 3.19% latency, 343.65 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 971.79 us, 0.20% latency, 1.24 TFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 710.49 us, 0.15% latency, 1.7 TFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 708.34 us, 0.15% latency, 1.71 TFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 93.22 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.26 ms, 0.47% latency, 535.06 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.66 ms, 0.35% latency, 726.91 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 244.38 us, 0.05% latency, 16.09 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.39 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.73 ms, 0.78% latency, 1.3 TFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.13 ms, 0.65% latency, 1.54 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 522.85 us, 0.11% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 11.68 ms, 2.44% latency, 414.19 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 11.12 ms, 2.32% latency, 434.67 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 231.98 us, 0.05% latency, 16.95 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.34 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (7): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 47.34 ms, 9.90% latency, 340.51 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 17.54 ms, 3.67% latency, 367.91 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.6 ms, 2.84% latency, 385.4 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.62 ms, 0.34% latency, 745.96 GFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.7 ms, 0.36% latency, 711.09 GFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.56 ms, 0.33% latency, 773.4 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 55.07 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.86 ms, 0.81% latency, 313.61 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.28 ms, 0.69% latency, 368.53 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 244.14 us, 0.05% latency, 16.11 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.15 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.66 ms, 1.18% latency, 853.39 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.08 ms, 1.06% latency, 951.02 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 520.47 us, 0.11% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.98 ms, 5.01% latency, 201.68 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.47 ms, 4.91% latency, 205.89 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 212.19 us, 0.04% latency, 18.53 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.34 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (8): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 49.81 ms, 10.41% latency, 323.62 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 20.53 ms, 4.29% latency, 314.32 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 16.79 ms, 3.51% latency, 312.13 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.82 ms, 0.59% latency, 428.24 GFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.84 ms, 0.59% latency, 424.76 GFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.83 ms, 0.59% latency, 426.62 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 55.07 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.67 ms, 0.77% latency, 330.45 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.19 ms, 0.67% latency, 378.72 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 181.44 us, 0.04% latency, 21.67 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 29.56 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.78 ms, 1.21% latency, 836.06 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.34 ms, 1.12% latency, 904.01 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 372.65 us, 0.08% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.33 ms, 4.88% latency, 207.3 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.81 ms, 4.77% latency, 211.82 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 224.59 us, 0.05% latency, 17.51 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 59.84 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (9): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 57.46 ms, 12.01% latency, 280.56 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 26.35 ms, 5.51% latency, 244.85 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 23.33 ms, 4.88% latency, 224.66 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 8.89 ms, 1.86% latency, 135.81 GFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.81 ms, 0.59% latency, 429.37 GFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.89 ms, 0.60% latency, 418.27 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 106.33 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.94 ms, 0.61% latency, 412.05 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.33 ms, 0.49% latency, 519.17 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 324.49 us, 0.07% latency, 12.12 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.72 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.91 ms, 1.24% latency, 817.32 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.38 ms, 1.13% latency, 897.57 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 455.86 us, 0.10% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 24.89 ms, 5.20% latency, 194.32 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 24.27 ms, 5.07% latency, 199.11 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 274.9 us, 0.06% latency, 14.3 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 47.68 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (10): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 51.0 ms, 10.66% latency, 316.07 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 21.07 ms, 4.41% latency, 306.2 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 17.31 ms, 3.62% latency, 302.72 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.08 ms, 0.64% latency, 392.12 GFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.18 ms, 0.66% latency, 380.34 GFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.76 ms, 0.58% latency, 437.87 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 59.6 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.68 ms, 0.77% latency, 329.45 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.14 ms, 0.66% latency, 385.23 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 239.61 us, 0.05% latency, 16.41 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 41.48 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.66 ms, 1.18% latency, 853.96 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.14 ms, 1.08% latency, 939.47 GFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 443.46 us, 0.09% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 24.08 ms, 5.03% latency, 200.82 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.49 ms, 4.91% latency, 205.71 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 251.05 us, 0.05% latency, 15.66 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.62 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
                (11): BertLayer(
                  7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 50.0 ms, 10.45% latency, 322.43 GFLOPS, 
                  (attention): BertAttention(
                    2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 21.11 ms, 4.41% latency, 305.74 GFLOPS, 
                    (self): BertSelfAttention(
                      1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 16.88 ms, 3.53% latency, 310.43 GFLOPS, 
                      (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 3.07 ms, 0.64% latency, 393.98 GFLOPS, in_features=768, out_features=768, bias=True)
                      (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.76 ms, 0.58% latency, 437.3 GFLOPS, in_features=768, out_features=768, bias=True)
                      (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.57 ms, 0.54% latency, 470.0 GFLOPS, in_features=768, out_features=768, bias=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 58.65 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                    (output): BertSelfOutput(
                      592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 4.14 ms, 0.87% latency, 292.53 GFLOPS, 
                      (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.92 ms, 0.61% latency, 413.02 GFLOPS, in_features=768, out_features=768, bias=True)
                      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 239.85 us, 0.05% latency, 16.39 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.43 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                    )
                  )
                  (intermediate): BertIntermediate(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 5.19 ms, 1.08% latency, 931.86 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.71 ms, 0.98% latency, 1.03 TFLOPS, in_features=768, out_features=3072, bias=True)
                    (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 407.93 us, 0.09% latency, 0.0 FLOPS, )
                  )
                  (output): BertOutput(
                    2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 23.43 ms, 4.90% latency, 206.43 GFLOPS, 
                    (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 22.82 ms, 4.77% latency, 211.75 GFLOPS, in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 264.88 us, 0.06% latency, 14.84 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 47.45 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
                  )
                )
              )
            )
            (pooler): BertPooler(
              590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 2.92 ms, 0.61% latency, 806.88 MFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 2.75 ms, 0.57% latency, 858.55 MFLOPS, in_features=768, out_features=768, bias=True)
              (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 61.75 us, 0.01% latency, 0.0 FLOPS, )
            )
          )
          (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 30.99 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          (classifier): Linear(1.54 k, 0.00% Params, 3.07 KMACs, 0.00% MACs, 66.28 us, 0.01% latency, 92.7 MFLOPS, in_features=768, out_features=2, bias=True)
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
