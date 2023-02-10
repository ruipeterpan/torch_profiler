
class Profile(object):
    def __init__(
        self, 
        name: str, 
        type: str, 
        depth: int,
        num_params: int,
        input_shape: list,
        output_shape: list,
        input_elem_bytes: int,
        output_elem_bytes: int,
        fwd_latency: float,
        macs: float,
        fwd_flops: float,
    ):  # each Profile corresponds to the profile of a torch.nn module
        self.name = name  # torch.nn module name
        self.type = type  # torch.nn module type
        self.depth = depth  # depth of current module in model
        self.num_params = num_params  # number of parameters
        self.num_params_pctg = None  # percentage of total params
        # shape of input/output tensor
        self.input_shape = input_shape
        self.output_shape = output_shape
        # size in bytes of an individual element in the input/output tensor
        self.input_elem_bytes = input_elem_bytes
        self.output_elem_bytes = output_elem_bytes
        self.fwd_latency = fwd_latency  # fwd latency (forward propagation latency) in ms
        self.fwd_latency_pctg = None  # percentage of total fwd latency 
        # NOTE(ruipan): the following aren't matching with the
        # original deepspeed profiler's output... fix later if needed
        self.macs = macs  # number of multiply-accumulate operations (MACs)
        self.macs_pctg = None  # percentage of total MACs
        # number of floating-point operations (flops) OR floating-point operations per second (FLOPS)??
        self.fwd_flops = fwd_flops
        self.fwd_flops_pctg = None
        self.children = []
    
    def set_child_modules(self, children) -> None:
        """Sets up the child modules, and fills in 
        the overall percentage statistics

        Args:
            children (Profile): profile of child of module
            of which the current profile is from
        """
        self.children = children
        if self.name == "model":  # outermost model
            total_duration = self.fwd_latency
            self.calculate_overall_stats(total_duration=total_duration)

    def __str__(self) -> str:
        # indent = "├─" * self.depth
        indent = "\t" * self.depth
        curr_str = (f"{indent}({self.name}): {self.type}, num_params {self.num_params}, " \
            f"{round(self.fwd_latency, 3)}ms, {round(self.fwd_latency_pctg, 3)}% latency, " \
            f"input shape {self.input_shape}, output shape {self.output_shape}\n")
        for child in self.children:
            curr_str += str(child)
        return curr_str
    
    def calculate_overall_stats(self, total_duration: float) -> None:
        """Recursively fills in the overall percentage statistics

        Args:
            total_duration (float): total duration of one fwd
            pass of the model in ms
        """
        if self.type == "ModuleList":  # latency is 0, aggregate latencies from children first
            self.fwd_latency = sum([c.fwd_latency for c in self.children])

        self.fwd_latency_pctg = 100 * self.fwd_latency / total_duration
        for child in self.children:
            child.calculate_overall_stats(total_duration=total_duration)
