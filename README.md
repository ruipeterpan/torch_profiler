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
(model): ResNet, num_params 11689512, 94.485ms, 100.0% latency, input shape [16, 3, 224, 224], output shape [16, 1000]
        (conv1): Conv2d, num_params 9408, 15.642ms, 16.555% latency, input shape [16, 3, 224, 224], output shape [16, 64, 112, 112]
        (bn1): BatchNorm2d, num_params 128, 5.027ms, 5.32% latency, input shape [16, 64, 112, 112], output shape [16, 64, 112, 112]
        (relu): ReLU, num_params 0, 0.911ms, 0.964% latency, input shape [16, 64, 112, 112], output shape [16, 64, 112, 112]
        (maxpool): MaxPool2d, num_params 0, 4.476ms, 4.738% latency, input shape [16, 64, 112, 112], output shape [16, 64, 56, 56]
        (layer1): Sequential, num_params 147968, 22.401ms, 23.708% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                (0): BasicBlock, num_params 73984, 11.021ms, 11.664% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv1): Conv2d, num_params 36864, 4.471ms, 4.732% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn1): BatchNorm2d, num_params 128, 0.898ms, 0.951% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (relu): ReLU, num_params 0, 0.362ms, 0.384% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv2): Conv2d, num_params 36864, 3.85ms, 4.074% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn2): BatchNorm2d, num_params 128, 0.888ms, 0.939% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                (1): BasicBlock, num_params 73984, 11.271ms, 11.929% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv1): Conv2d, num_params 36864, 3.817ms, 4.04% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn1): BatchNorm2d, num_params 128, 1.507ms, 1.595% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (relu): ReLU, num_params 0, 0.442ms, 0.468% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (conv2): Conv2d, num_params 36864, 3.981ms, 4.214% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
                        (bn2): BatchNorm2d, num_params 128, 1.017ms, 1.077% latency, input shape [16, 64, 56, 56], output shape [16, 64, 56, 56]
        (layer2): Sequential, num_params 525568, 15.624ms, 16.536% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                (0): BasicBlock, num_params 230144, 9.375ms, 9.923% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                        (conv1): Conv2d, num_params 73728, 2.725ms, 2.884% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                        (bn1): BatchNorm2d, num_params 256, 0.545ms, 0.577% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (relu): ReLU, num_params 0, 0.204ms, 0.216% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv2): Conv2d, num_params 147456, 2.508ms, 2.654% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn2): BatchNorm2d, num_params 256, 0.46ms, 0.487% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (downsample): Sequential, num_params 8448, 2.631ms, 2.785% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                                (0): Conv2d, num_params 8192, 2.038ms, 2.156% latency, input shape [16, 64, 56, 56], output shape [16, 128, 28, 28]
                                (1): BatchNorm2d, num_params 256, 0.501ms, 0.53% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                (1): BasicBlock, num_params 295424, 6.164ms, 6.524% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv1): Conv2d, num_params 147456, 1.831ms, 1.938% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn1): BatchNorm2d, num_params 256, 0.616ms, 0.652% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (relu): ReLU, num_params 0, 0.205ms, 0.217% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (conv2): Conv2d, num_params 147456, 2.761ms, 2.922% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
                        (bn2): BatchNorm2d, num_params 256, 0.48ms, 0.508% latency, input shape [16, 128, 28, 28], output shape [16, 128, 28, 28]
        (layer3): Sequential, num_params 2099712, 14.438ms, 15.281% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                (0): BasicBlock, num_params 919040, 8.039ms, 8.509% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                        (conv1): Conv2d, num_params 294912, 2.385ms, 2.524% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                        (bn1): BatchNorm2d, num_params 512, 0.33ms, 0.349% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (relu): ReLU, num_params 0, 0.195ms, 0.206% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv2): Conv2d, num_params 589824, 2.326ms, 2.462% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn2): BatchNorm2d, num_params 512, 0.341ms, 0.361% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (downsample): Sequential, num_params 33280, 2.147ms, 2.273% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                                (0): Conv2d, num_params 32768, 1.697ms, 1.796% latency, input shape [16, 128, 28, 28], output shape [16, 256, 14, 14]
                                (1): BatchNorm2d, num_params 512, 0.369ms, 0.39% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                (1): BasicBlock, num_params 1180672, 6.317ms, 6.685% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv1): Conv2d, num_params 589824, 2.217ms, 2.346% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn1): BatchNorm2d, num_params 512, 0.447ms, 0.473% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (relu): ReLU, num_params 0, 0.196ms, 0.208% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (conv2): Conv2d, num_params 589824, 2.577ms, 2.727% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
                        (bn2): BatchNorm2d, num_params 512, 0.597ms, 0.632% latency, input shape [16, 256, 14, 14], output shape [16, 256, 14, 14]
        (layer4): Sequential, num_params 8393728, 14.592ms, 15.444% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                (0): BasicBlock, num_params 3673088, 8.246ms, 8.727% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                        (conv1): Conv2d, num_params 1179648, 2.546ms, 2.695% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                        (bn1): BatchNorm2d, num_params 1024, 0.281ms, 0.298% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (relu): ReLU, num_params 0, 0.203ms, 0.215% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv2): Conv2d, num_params 2359296, 3.17ms, 3.356% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn2): BatchNorm2d, num_params 1024, 0.283ms, 0.3% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (downsample): Sequential, num_params 132096, 1.485ms, 1.572% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                                (0): Conv2d, num_params 131072, 1.112ms, 1.177% latency, input shape [16, 256, 14, 14], output shape [16, 512, 7, 7]
                                (1): BatchNorm2d, num_params 1024, 0.283ms, 0.299% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                (1): BasicBlock, num_params 4720640, 6.264ms, 6.63% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv1): Conv2d, num_params 2359296, 2.629ms, 2.783% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn1): BatchNorm2d, num_params 1024, 0.25ms, 0.264% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (relu): ReLU, num_params 0, 0.189ms, 0.2% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (conv2): Conv2d, num_params 2359296, 2.697ms, 2.855% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
                        (bn2): BatchNorm2d, num_params 1024, 0.262ms, 0.278% latency, input shape [16, 512, 7, 7], output shape [16, 512, 7, 7]
        (avgpool): AdaptiveAvgPool2d, num_params 0, 0.407ms, 0.43% latency, input shape [16, 512, 7, 7], output shape [16, 512, 1, 1]
        (fc): Linear, num_params 513000, 0.502ms, 0.531% latency, input shape [16, 512], output shape [16, 1000]
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
fwd latency:                                                  76.73 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          758.27 GFLOPS

----------------------------- Aggregated Profile per GPU -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'ResNet': '11.69 M'}
    MACs        - {'ResNet': '29.03 GMACs'}
    fwd latency - {'ResNet': '76.73 ms'}
depth 1:
    params      - {'Sequential': '11.17 M'}
    MACs        - {'Sequential': '27.13 GMACs'}
    fwd latency - {'Sequential': '58.92 ms'}
depth 2:
    params      - {'BasicBlock': '11.17 M'}
    MACs        - {'BasicBlock': '27.13 GMACs'}
    fwd latency - {'BasicBlock': '58.66 ms'}
depth 3:
    params      - {'Conv2d': '10.99 M'}
    MACs        - {'Conv2d': '26.82 GMACs'}
    fwd latency - {'Conv2d': '44.56 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

ResNet(
  11.69 M, 100.00% Params, 29.03 GMACs, 100.00% MACs, 76.73 ms, 100.00% latency, 758.27 GFLOPS, 
  (conv1): Conv2d(9.41 k, 0.08% Params, 1.89 GMACs, 6.51% MACs, 10.17 ms, 13.25% latency, 371.35 GFLOPS, 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 3.31 ms, 4.31% latency, 7.77 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 715.02 us, 0.93% latency, 17.96 GFLOPS, inplace=True)
  (maxpool): MaxPool2d(0, 0.00% Params, 0 MACs, 0.00% MACs, 2.64 ms, 3.44% latency, 4.87 GFLOPS, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    147.97 k, 1.27% Params, 7.4 GMACs, 25.49% MACs, 22.56 ms, 29.40% latency, 657.61 GFLOPS, 
    (0): BasicBlock(
      73.98 k, 0.63% Params, 3.7 GMACs, 12.75% MACs, 11.39 ms, 14.84% latency, 651.51 GFLOPS, 
      (conv1): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 4.06 ms, 5.29% latency, 911.6 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 1.04 ms, 1.35% latency, 6.18 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 396.73 us, 0.52% latency, 16.19 GFLOPS, inplace=True)
      (conv2): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 4.26 ms, 5.55% latency, 869.02 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 1.06 ms, 1.38% latency, 6.07 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      73.98 k, 0.63% Params, 3.7 GMACs, 12.75% MACs, 11.11 ms, 14.48% latency, 667.81 GFLOPS, 
      (conv1): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 4.19 ms, 5.47% latency, 882.21 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 558.38 us, 0.73% latency, 11.5 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 372.17 us, 0.49% latency, 17.26 GFLOPS, inplace=True)
      (conv2): Conv2d(36.86 k, 0.32% Params, 1.85 GMACs, 6.37% MACs, 4.21 ms, 5.48% latency, 879.61 GFLOPS, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, 0.00% Params, 0 MACs, 0.00% MACs, 1.04 ms, 1.36% latency, 6.15 GFLOPS, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    525.57 k, 4.50% Params, 6.58 GMACs, 22.66% MACs, 13.32 ms, 17.36% latency, 989.34 GFLOPS, 
    (0): BasicBlock(
      230.14 k, 1.97% Params, 2.88 GMACs, 9.91% MACs, 8.28 ms, 10.79% latency, 696.87 GFLOPS, 
      (conv1): Conv2d(73.73 k, 0.63% Params, 924.84 MMACs, 3.19% MACs, 3.26 ms, 4.25% latency, 567.24 GFLOPS, 64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 190.02 us, 0.25% latency, 16.9 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 209.33 us, 0.27% latency, 15.34 GFLOPS, inplace=True)
      (conv2): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 2.34 ms, 3.05% latency, 1.58 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 133.75 us, 0.17% latency, 24.01 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        8.45 k, 0.07% Params, 102.76 MMACs, 0.35% MACs, 1.79 ms, 2.34% latency, 116.44 GFLOPS, 
        (0): Conv2d(8.19 k, 0.07% Params, 102.76 MMACs, 0.35% MACs, 1.57 ms, 2.05% latency, 130.73 GFLOPS, 64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 154.73 us, 0.20% latency, 20.75 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      295.42 k, 2.53% Params, 3.7 GMACs, 12.75% MACs, 4.98 ms, 6.49% latency, 1.49 TFLOPS, 
      (conv1): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 2.19 ms, 2.86% latency, 1.69 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 128.98 us, 0.17% latency, 24.9 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 186.92 us, 0.24% latency, 17.18 GFLOPS, inplace=True)
      (conv2): Conv2d(147.46 k, 1.26% Params, 1.85 GMACs, 6.37% MACs, 2.03 ms, 2.64% latency, 1.83 TFLOPS, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, 0.00% Params, 0 MACs, 0.00% MACs, 157.12 us, 0.20% latency, 20.44 GFLOPS, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    2.1 M, 17.96% Params, 6.58 GMACs, 22.66% MACs, 10.45 ms, 13.62% latency, 1.26 TFLOPS, 
    (0): BasicBlock(
      919.04 k, 7.86% Params, 2.88 GMACs, 9.91% MACs, 5.77 ms, 7.52% latency, 998.82 GFLOPS, 
      (conv1): Conv2d(294.91 k, 2.52% Params, 924.84 MMACs, 3.19% MACs, 1.73 ms, 2.25% latency, 1.07 TFLOPS, 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 125.41 us, 0.16% latency, 12.8 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 232.93 us, 0.30% latency, 6.89 GFLOPS, inplace=True)
      (conv2): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 2.17 ms, 2.83% latency, 1.7 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 108.24 us, 0.14% latency, 14.83 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        33.28 k, 0.28% Params, 102.76 MMACs, 0.35% MACs, 1.07 ms, 1.39% latency, 193.87 GFLOPS, 
        (0): Conv2d(32.77 k, 0.28% Params, 102.76 MMACs, 0.35% MACs, 895.26 us, 1.17% latency, 229.57 GFLOPS, 128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 109.91 us, 0.14% latency, 14.61 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      1.18 M, 10.10% Params, 3.7 GMACs, 12.75% MACs, 4.62 ms, 6.03% latency, 1.6 TFLOPS, 
      (conv1): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 2.06 ms, 2.68% latency, 1.8 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 104.43 us, 0.14% latency, 15.38 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 185.01 us, 0.24% latency, 8.68 GFLOPS, inplace=True)
      (conv2): Conv2d(589.82 k, 5.05% Params, 1.85 GMACs, 6.37% MACs, 1.89 ms, 2.46% latency, 1.96 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, 0.00% Params, 0 MACs, 0.00% MACs, 98.94 us, 0.13% latency, 16.23 GFLOPS, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    8.39 M, 71.81% Params, 6.58 GMACs, 22.66% MACs, 12.59 ms, 16.41% latency, 1.04 TFLOPS, 
    (0): BasicBlock(
      3.67 M, 31.42% Params, 2.88 GMACs, 9.91% MACs, 6.37 ms, 8.30% latency, 903.78 GFLOPS, 
      (conv1): Conv2d(1.18 M, 10.09% Params, 924.84 MMACs, 3.19% MACs, 1.91 ms, 2.48% latency, 970.5 GFLOPS, 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 135.42 us, 0.18% latency, 5.93 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 168.8 us, 0.22% latency, 4.76 GFLOPS, inplace=True)
      (conv2): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.92 ms, 3.81% latency, 1.27 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 107.29 us, 0.14% latency, 7.48 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        132.1 k, 1.13% Params, 102.76 MMACs, 0.35% MACs, 875.95 us, 1.14% latency, 235.54 GFLOPS, 
        (0): Conv2d(131.07 k, 1.12% Params, 102.76 MMACs, 0.35% MACs, 699.04 us, 0.91% latency, 294.0 GFLOPS, 256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 106.57 us, 0.14% latency, 7.53 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      4.72 M, 40.38% Params, 3.7 GMACs, 12.75% MACs, 6.15 ms, 8.01% latency, 1.2 TFLOPS, 
      (conv1): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.63 ms, 3.42% latency, 1.41 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 106.81 us, 0.14% latency, 7.52 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0, 0.00% Params, 0 MACs, 0.00% MACs, 239.13 us, 0.31% latency, 3.36 GFLOPS, inplace=True)
      (conv2): Conv2d(2.36 M, 20.18% Params, 1.85 GMACs, 6.37% MACs, 2.72 ms, 3.55% latency, 1.36 TFLOPS, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(1.02 k, 0.01% Params, 0 MACs, 0.00% MACs, 164.99 us, 0.22% latency, 4.87 GFLOPS, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 0 MACs, 0.00% MACs, 229.36 us, 0.30% latency, 1.75 GFLOPS, output_size=(1, 1))
  (fc): Linear(513.0 k, 4.39% Params, 8.19 MMACs, 0.03% MACs, 335.45 us, 0.44% latency, 48.84 GFLOPS, in_features=512, out_features=1000, bias=True)
)
------------------------------------------------------------------------------

  ```
</details>


### BERT example

<details>
  <summary>Profiler output</summary>

  ```bash
$ python3 example_bert.py
(model): BertForSequenceClassification, num_params 109483778, 315.584ms, 100.0% latency, input shape [], output shape [2, 2]
        (bert): BertModel, num_params 109482240, 313.903ms, 99.467% latency, input shape [2, 512], output shape [2, 768]
                (embeddings): BertEmbeddings, num_params 23837184, 5.712ms, 1.81% latency, input shape [], output shape [2, 512, 768]
                        (word_embeddings): Embedding, num_params 23440896, 2.5ms, 0.792% latency, input shape [2, 512], output shape [2, 512, 768]
                        (position_embeddings): Embedding, num_params 393216, 0.384ms, 0.122% latency, input shape [1, 512], output shape [1, 512, 768]
                        (token_type_embeddings): Embedding, num_params 1536, 0.389ms, 0.123% latency, input shape [2, 512], output shape [2, 512, 768]
                        (LayerNorm): LayerNorm, num_params 1536, 0.88ms, 0.279% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                        (dropout): Dropout, num_params 0, 0.337ms, 0.107% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                (encoder): BertEncoder, num_params 85054464, 306.437ms, 97.102% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                        (layer): ModuleList, num_params 85054464, 305.364ms, 96.762% latency, input shape None, output shape None
                                (0): BertLayer, num_params 7087872, 32.938ms, 10.437% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 18.835ms, 5.968% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 16.574ms, 5.252% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.801ms, 0.571% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 1.282ms, 0.406% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 1.224ms, 0.388% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.122ms, 0.039% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.061ms, 0.653% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.263ms, 0.4% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.337ms, 0.107% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.079ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 8.577ms, 2.718% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 2.989ms, 0.947% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 5.406ms, 1.713% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.773ms, 1.512% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.921ms, 1.242% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.376ms, 0.119% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.082ms, 0.026% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (1): BertLayer, num_params 7087872, 21.594ms, 6.842% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 13.891ms, 4.402% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 11.753ms, 3.724% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.061ms, 0.336% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.863ms, 0.273% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.836ms, 0.265% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.115ms, 0.037% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.002ms, 0.634% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.146ms, 0.363% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.401ms, 0.127% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.086ms, 0.027% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.405ms, 1.079% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 2.816ms, 0.892% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.463ms, 0.147% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.034ms, 1.278% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.286ms, 1.041% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.348ms, 0.11% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.08ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (2): BertLayer, num_params 7087872, 27.773ms, 8.801% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 19.182ms, 6.078% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 16.907ms, 5.357% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 0.955ms, 0.303% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.828ms, 0.262% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.837ms, 0.265% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.113ms, 0.036% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.152ms, 0.682% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.292ms, 0.409% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.376ms, 0.119% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.081ms, 0.026% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.928ms, 1.245% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.087ms, 0.978% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.709ms, 0.225% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.395ms, 1.393% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.597ms, 1.14% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.309ms, 0.098% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.08ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (3): BertLayer, num_params 7087872, 23.726ms, 7.518% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.232ms, 4.826% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.968ms, 4.109% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.2ms, 0.38% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.949ms, 0.301% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.92ms, 0.291% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.123ms, 0.039% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.135ms, 0.676% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.271ms, 0.403% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.35ms, 0.111% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.076ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 3.844ms, 1.218% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.025ms, 0.958% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.696ms, 0.22% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.381ms, 1.388% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.562ms, 1.129% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.322ms, 0.102% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.094ms, 0.03% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (4): BertLayer, num_params 7087872, 24.349ms, 7.715% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.118ms, 4.79% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.685ms, 4.02% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.223ms, 0.388% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.968ms, 0.307% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.944ms, 0.299% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.122ms, 0.039% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.308ms, 0.731% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.48ms, 0.469% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.333ms, 0.106% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.078ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.471ms, 1.417% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.678ms, 1.165% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.669ms, 0.212% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.484ms, 1.421% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.712ms, 1.176% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.322ms, 0.102% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.083ms, 0.026% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (5): BertLayer, num_params 7087872, 24.154ms, 7.654% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.06ms, 4.772% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.672ms, 4.015% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.21ms, 0.383% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.941ms, 0.298% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.977ms, 0.31% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.116ms, 0.037% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.26ms, 0.716% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.427ms, 0.452% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.347ms, 0.11% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.077ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.27ms, 1.353% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.491ms, 1.106% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.663ms, 0.21% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.555ms, 1.444% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.763ms, 1.192% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.337ms, 0.107% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.083ms, 0.026% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (6): BertLayer, num_params 7087872, 23.972ms, 7.596% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.142ms, 4.798% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.743ms, 4.038% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.214ms, 0.385% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.991ms, 0.314% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.921ms, 0.292% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.117ms, 0.037% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.269ms, 0.719% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.436ms, 0.455% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.315ms, 0.1% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.083ms, 0.026% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.147ms, 1.314% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.41ms, 1.081% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.623ms, 0.197% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.385ms, 1.39% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.605ms, 1.142% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.334ms, 0.106% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.08ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (7): BertLayer, num_params 7087872, 25.682ms, 8.138% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.514ms, 4.916% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 12.894ms, 4.086% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.2ms, 0.38% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.946ms, 0.3% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.967ms, 0.306% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.117ms, 0.037% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.495ms, 0.791% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.601ms, 0.507% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.361ms, 0.115% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.087ms, 0.028% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.638ms, 1.47% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.801ms, 1.204% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.707ms, 0.224% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 5.265ms, 1.668% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 4.463ms, 1.414% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.326ms, 0.103% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.07ms, 0.022% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (8): BertLayer, num_params 7087872, 25.846ms, 8.19% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 16.32ms, 5.171% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 13.755ms, 4.359% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.379ms, 0.437% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 1.0ms, 0.317% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.955ms, 0.303% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.122ms, 0.039% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.437ms, 0.772% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.585ms, 0.502% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.32ms, 0.101% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.077ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.637ms, 1.469% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.82ms, 1.211% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.7ms, 0.222% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.624ms, 1.465% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.83ms, 1.214% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.311ms, 0.099% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.065ms, 0.021% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (9): BertLayer, num_params 7087872, 25.207ms, 7.987% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.782ms, 5.001% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 13.32ms, 4.221% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.251ms, 0.397% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.955ms, 0.303% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.988ms, 0.313% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.121ms, 0.038% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.333ms, 0.739% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.469ms, 0.465% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.333ms, 0.106% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.088ms, 0.028% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.586ms, 1.453% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.806ms, 1.206% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.661ms, 0.209% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.559ms, 1.445% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.772ms, 1.195% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.331ms, 0.105% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.076ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (10): BertLayer, num_params 7087872, 25.231ms, 7.995% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.882ms, 5.033% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 13.411ms, 4.249% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.235ms, 0.391% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.947ms, 0.3% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.918ms, 0.291% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.119ms, 0.038% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.336ms, 0.74% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.504ms, 0.477% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.336ms, 0.107% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.077ms, 0.024% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.444ms, 1.408% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.681ms, 1.167% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.646ms, 0.205% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.64ms, 1.47% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.849ms, 1.22% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.332ms, 0.105% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.078ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                (11): BertLayer, num_params 7087872, 24.892ms, 7.887% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (attention): BertAttention, num_params 2363904, 15.747ms, 4.99% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (self): BertSelfAttention, num_params 1771776, 13.273ms, 4.206% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (query): Linear, num_params 590592, 1.275ms, 0.404% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (key): Linear, num_params 590592, 0.971ms, 0.308% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (value): Linear, num_params 590592, 0.94ms, 0.298% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.119ms, 0.038% latency, input shape [2, 12, 512, 512], output shape [2, 12, 512, 512]
                                                (output): BertSelfOutput, num_params 592128, 2.346ms, 0.743% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dense): Linear, num_params 590592, 1.478ms, 0.468% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (LayerNorm): LayerNorm, num_params 1536, 0.328ms, 0.104% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                        (dropout): Dropout, num_params 0, 0.084ms, 0.027% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                        (intermediate): BertIntermediate, num_params 2362368, 4.357ms, 1.381% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (dense): Linear, num_params 2362368, 3.585ms, 1.136% latency, input shape [2, 512, 768], output shape [2, 512, 3072]
                                                (intermediate_act_fn): GELUActivation, num_params 0, 0.652ms, 0.207% latency, input shape [2, 512, 3072], output shape [2, 512, 3072]
                                        (output): BertOutput, num_params 2361600, 4.523ms, 1.433% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (dense): Linear, num_params 2360064, 3.708ms, 1.175% latency, input shape [2, 512, 3072], output shape [2, 512, 768]
                                                (LayerNorm): LayerNorm, num_params 1536, 0.337ms, 0.107% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                                                (dropout): Dropout, num_params 0, 0.077ms, 0.025% latency, input shape [2, 512, 768], output shape [2, 512, 768]
                (pooler): BertPooler, num_params 590592, 0.672ms, 0.213% latency, input shape [2, 512, 768], output shape [2, 768]
                        (dense): Linear, num_params 590592, 0.216ms, 0.068% latency, input shape [2, 768], output shape [2, 768]
                        (activation): Tanh, num_params 0, 0.272ms, 0.086% latency, input shape [2, 768], output shape [2, 768]
        (dropout): Dropout, num_params 0, 0.066ms, 0.021% latency, input shape [2, 768], output shape [2, 768]
        (classifier): Linear, num_params 1538, 0.104ms, 0.033% latency, input shape [2, 768], output shape [2, 2]

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
fwd latency:                                                  297.4 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          650.47 GFLOPS

----------------------------- Aggregated Profile per GPU -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'BertForSequenceClassification': '109.48 M'}
    MACs        - {'BertForSequenceClassification': '96.64 GMACs'}
    fwd latency - {'BertForSequenceClassification': '297.4 ms'}
depth 1:
    params      - {'BertModel': '109.48 M'}
    MACs        - {'BertModel': '96.64 GMACs'}
    fwd latency - {'BertModel': '296.95 ms'}
depth 2:
    params      - {'BertEncoder': '85.05 M'}
    MACs        - {'BertEncoder': '96.64 GMACs'}
    fwd latency - {'BertEncoder': '294.59 ms'}
depth 3:
    params      - {'ModuleList': '85.05 M'}
    MACs        - {'ModuleList': '96.64 GMACs'}
    fwd latency - {'ModuleList': '294.23 ms'}
depth 4:
    params      - {'BertLayer': '85.05 M'}
    MACs        - {'BertLayer': '96.64 GMACs'}
    fwd latency - {'BertLayer': '294.23 ms'}
depth 5:
    params      - {'BertAttention': '28.37 M'}
    MACs        - {'BertAttention': '38.65 GMACs'}
    fwd latency - {'BertAttention': '199.37 ms'}
depth 6:
    params      - {'Linear': '56.67 M'}
    MACs        - {'Linear': '57.98 GMACs'}
    fwd latency - {'BertSelfAttention': '175.64 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order: 
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

BertForSequenceClassification(
  109.48 M, 100.00% Params, 96.64 GMACs, 100.00% MACs, 297.4 ms, 100.00% latency, 650.47 GFLOPS, 
  (bert): BertModel(
    109.48 M, 100.00% Params, 96.64 GMACs, 100.00% MACs, 296.95 ms, 99.85% latency, 651.46 GFLOPS, 
    (embeddings): BertEmbeddings(
      23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.63 ms, 0.55% latency, 2.41 GFLOPS, 
      (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 672.82 us, 0.23% latency, 0.0 FLOPS, 30522, 768, padding_idx=0)
      (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 156.16 us, 0.05% latency, 0.0 FLOPS, 512, 768)
      (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 131.85 us, 0.04% latency, 0.0 FLOPS, 2, 768)
      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 242.95 us, 0.08% latency, 16.19 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.48 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      85.05 M, 77.69% Params, 96.64 GMACs, 100.00% MACs, 294.59 ms, 99.06% latency, 656.65 GFLOPS, 
      (layer): ModuleList(
        85.05 M, 77.69% Params, 96.64 GMACs, 100.00% MACs, 294.23 ms, 98.94% latency, 657.45 GFLOPS, 
        (0): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 34.89 ms, 11.73% latency, 462.08 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 26.5 ms, 8.91% latency, 243.52 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 24.42 ms, 8.21% latency, 214.58 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 883.34 us, 0.30% latency, 1.37 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 720.02 us, 0.24% latency, 1.68 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 716.45 us, 0.24% latency, 1.69 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 61.99 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.0 ms, 0.67% latency, 606.79 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.26 ms, 0.42% latency, 960.12 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 234.13 us, 0.08% latency, 16.79 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.1 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.16 ms, 1.40% latency, 1.16 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.47 ms, 1.17% latency, 1.39 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 624.18 us, 0.21% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.0 ms, 1.34% latency, 1.21 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.37 ms, 1.13% latency, 1.43 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 299.93 us, 0.10% latency, 13.11 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 34.33 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (1): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 23.32 ms, 7.84% latency, 691.22 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 15.4 ms, 5.18% latency, 418.94 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.38 ms, 4.50% latency, 391.57 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.17 ms, 0.39% latency, 1.03 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 863.79 us, 0.29% latency, 1.4 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 783.92 us, 0.26% latency, 1.54 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 57.7 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.94 ms, 0.65% latency, 623.23 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.3 ms, 0.44% latency, 928.62 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 254.39 us, 0.09% latency, 15.46 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.58 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.83 ms, 1.29% latency, 1.26 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.17 ms, 1.07% latency, 1.52 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 595.81 us, 0.20% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.91 ms, 1.31% latency, 1.24 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.31 ms, 1.11% latency, 1.46 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 274.18 us, 0.09% latency, 14.34 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 33.14 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (2): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 27.64 ms, 9.29% latency, 583.3 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 19.89 ms, 6.69% latency, 324.37 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 17.88 ms, 6.01% latency, 293.11 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.16 ms, 0.39% latency, 1.05 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 785.83 us, 0.26% latency, 1.54 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 765.8 us, 0.26% latency, 1.58 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 51.5 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.94 ms, 0.65% latency, 623.69 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.3 ms, 0.44% latency, 926.07 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 257.73 us, 0.09% latency, 15.26 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.1 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.7 ms, 1.24% latency, 1.31 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.01 ms, 1.01% latency, 1.61 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 622.75 us, 0.21% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.87 ms, 1.30% latency, 1.25 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.25 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 256.3 us, 0.09% latency, 15.34 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 35.05 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (3): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.4 ms, 7.53% latency, 719.51 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.67 ms, 4.93% latency, 439.78 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 12.61 ms, 4.24% latency, 415.76 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.08 ms, 0.36% latency, 1.12 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 773.19 us, 0.26% latency, 1.56 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 723.6 us, 0.24% latency, 1.67 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 59.13 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.98 ms, 0.66% latency, 612.86 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.33 ms, 0.45% latency, 908.64 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 267.51 us, 0.09% latency, 14.7 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.58 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.7 ms, 1.24% latency, 1.31 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.1 ms, 1.04% latency, 1.56 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 527.86 us, 0.18% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.86 ms, 1.30% latency, 1.25 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.22 ms, 1.08% latency, 1.5 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 282.76 us, 0.10% latency, 13.91 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.0 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (4): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.22 ms, 7.47% latency, 725.38 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.46 ms, 4.86% latency, 446.28 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 12.58 ms, 4.23% latency, 416.67 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.07 ms, 0.36% latency, 1.13 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 742.67 us, 0.25% latency, 1.63 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 724.55 us, 0.24% latency, 1.67 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.98 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.8 ms, 0.61% latency, 672.18 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.17 ms, 0.39% latency, 1.04 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 299.45 us, 0.10% latency, 13.13 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.91 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.74 ms, 1.26% latency, 1.29 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.06 ms, 1.03% latency, 1.58 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 595.57 us, 0.20% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.84 ms, 1.29% latency, 1.26 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.25 ms, 1.09% latency, 1.48 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 254.39 us, 0.09% latency, 15.46 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 34.57 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (5): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.04 ms, 7.41% latency, 731.45 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.35 ms, 4.82% latency, 449.71 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 12.48 ms, 4.20% latency, 419.85 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.15 ms, 0.39% latency, 1.05 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 761.99 us, 0.26% latency, 1.59 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 735.76 us, 0.25% latency, 1.64 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 54.84 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.79 ms, 0.60% latency, 675.76 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.21 ms, 0.41% latency, 994.61 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 221.01 us, 0.07% latency, 17.79 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.91 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.65 ms, 1.23% latency, 1.32 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.0 ms, 1.01% latency, 1.61 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 590.09 us, 0.20% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.86 ms, 1.30% latency, 1.25 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.24 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 257.97 us, 0.09% latency, 15.24 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.82 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (6): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.66 ms, 7.62% latency, 711.37 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.71 ms, 4.95% latency, 438.72 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 12.76 ms, 4.29% latency, 410.77 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.07 ms, 0.36% latency, 1.13 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 777.96 us, 0.26% latency, 1.55 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 759.84 us, 0.26% latency, 1.59 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.74 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.87 ms, 0.63% latency, 648.1 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.22 ms, 0.41% latency, 992.66 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 281.1 us, 0.09% latency, 13.99 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 41.72 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.94 ms, 1.32% latency, 1.23 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.26 ms, 1.10% latency, 1.48 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 615.6 us, 0.21% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.82 ms, 1.28% latency, 1.27 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.24 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 246.76 us, 0.08% latency, 15.93 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 35.05 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (7): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.94 ms, 7.72% latency, 702.57 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 15.15 ms, 5.10% latency, 425.84 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.15 ms, 4.42% latency, 398.52 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.07 ms, 0.36% latency, 1.13 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 757.69 us, 0.25% latency, 1.59 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 729.8 us, 0.25% latency, 1.66 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 61.04 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.92 ms, 0.64% latency, 632.22 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.27 ms, 0.43% latency, 948.97 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 279.19 us, 0.09% latency, 14.08 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.67 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.79 ms, 1.28% latency, 1.27 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.15 ms, 1.06% latency, 1.53 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 576.5 us, 0.19% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.82 ms, 1.28% latency, 1.27 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.23 ms, 1.08% latency, 1.5 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 264.17 us, 0.09% latency, 14.89 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.0 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (8): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 22.78 ms, 7.66% latency, 707.75 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 14.92 ms, 5.02% latency, 432.5 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.02 ms, 4.38% latency, 402.55 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.11 ms, 0.37% latency, 1.08 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 751.26 us, 0.25% latency, 1.61 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 747.68 us, 0.25% latency, 1.62 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.27 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.82 ms, 0.61% latency, 667.5 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.2 ms, 0.40% latency, 1.01 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 255.35 us, 0.09% latency, 15.4 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.62 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.83 ms, 1.29% latency, 1.26 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.17 ms, 1.07% latency, 1.52 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 588.89 us, 0.20% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.84 ms, 1.29% latency, 1.26 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.23 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 268.46 us, 0.09% latency, 14.65 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.24 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (9): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 24.22 ms, 8.14% latency, 665.58 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 15.23 ms, 5.12% latency, 423.8 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 13.04 ms, 4.39% latency, 401.84 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.13 ms, 0.38% latency, 1.07 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 770.81 us, 0.26% latency, 1.57 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 739.57 us, 0.25% latency, 1.63 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.98 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 2.11 ms, 0.71% latency, 575.66 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.34 ms, 0.45% latency, 900.72 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 308.51 us, 0.10% latency, 12.75 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 49.35 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.26 ms, 1.43% latency, 1.13 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.35 ms, 1.13% latency, 1.44 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 787.5 us, 0.26% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 4.49 ms, 1.51% latency, 1.08 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.5 ms, 1.18% latency, 1.38 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 408.17 us, 0.14% latency, 9.63 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 63.66 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (10): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 28.87 ms, 9.71% latency, 558.42 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 20.85 ms, 7.01% latency, 309.44 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 18.86 ms, 6.34% latency, 277.94 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.3 ms, 0.44% latency, 931.86 GFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.09 ms, 0.37% latency, 1.1 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 936.51 us, 0.31% latency, 1.29 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 90.12 us, 0.03% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.9 ms, 0.64% latency, 637.53 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.22 ms, 0.41% latency, 993.64 GFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 291.35 us, 0.10% latency, 13.5 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 40.53 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.9 ms, 1.31% latency, 1.24 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.34 ms, 1.12% latency, 1.45 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 492.33 us, 0.17% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.91 ms, 1.31% latency, 1.24 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.24 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 270.61 us, 0.09% latency, 14.53 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 42.92 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (11): BertLayer(
          7.09 M, 6.47% Params, 8.05 GMACs, 8.33% MACs, 20.25 ms, 6.81% latency, 796.03 GFLOPS, 
          (attention): BertAttention(
            2.36 M, 2.16% Params, 3.22 GMACs, 3.33% MACs, 13.24 ms, 4.45% latency, 487.53 GFLOPS, 
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 2.62 GMACs, 2.71% MACs, 11.46 ms, 3.85% latency, 457.18 GFLOPS, 
              (query): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 930.55 us, 0.31% latency, 1.3 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 775.58 us, 0.26% latency, 1.56 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 714.54 us, 0.24% latency, 1.69 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 61.27 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.69 ms, 0.57% latency, 716.53 GFLOPS, 
              (dense): Linear(590.59 k, 0.54% Params, 603.98 MMACs, 0.62% MACs, 1.13 ms, 0.38% latency, 1.07 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 219.82 us, 0.07% latency, 17.89 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 38.86 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.98 ms, 1.00% latency, 1.62 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 2.49 ms, 0.84% latency, 1.94 TFLOPS, in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 422.24 us, 0.14% latency, 0.0 FLOPS, )
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.84 ms, 1.29% latency, 1.26 TFLOPS, 
            (dense): Linear(2.36 M, 2.16% Params, 2.42 GMACs, 2.50% MACs, 3.25 ms, 1.09% latency, 1.49 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 277.04 us, 0.09% latency, 14.19 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.43 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 319.0 us, 0.11% latency, 7.4 GFLOPS, 
      (dense): Linear(590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 144.0 us, 0.05% latency, 16.38 GFLOPS, in_features=768, out_features=768, bias=True)
      (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 52.21 us, 0.02% latency, 0.0 FLOPS, )
    )
  )
  (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 34.09 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
  (classifier): Linear(1.54 k, 0.00% Params, 3.07 KMACs, 0.00% MACs, 54.6 us, 0.02% latency, 112.53 MFLOPS, in_features=768, out_features=2, bias=True)
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
