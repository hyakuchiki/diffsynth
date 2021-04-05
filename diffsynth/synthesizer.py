import torch
import torch.nn as nn
from diffsynth.processor import Gen
import diffsynth.util as util

SCALE_FNS = {
    'sigmoid': lambda x, low, high: torch.sigmoid(x)*(high-low) + low,
    'freq_sigmoid': lambda x, low, high: util.frequencies_sigmoid(x, low, high),
    'exp_sigmoid': lambda x, low, high: util.exp_sigmoid(x, 10.0, high, 1e-7+low),
}

class Synthesizer(nn.Module):
    """
    defined by a DAG of processors in a similar manner to DDSP
    """

    def __init__(self, dag, name='synth', fixed_params={}):
        """
        
        Args:
            dag (list of tuples): [(processor, {'param_name':'INPUT_KEY' or 'processor.name'})] 
            ex.)    [   
                    (additive, {'amplitudes':'ADD_AMP', 'harmonic_distribution':'ADD_HARMONIC', 'f0_hz':'ADD_F0'}),
                    (filter, {'input':'additive', 'cutoff':'CUTOFF'}),
                    ...
                    ]
            name (str, optional): Defaults to 'synth'.
            fixed_params: Values of fixed parameters ex.) {'INPUT_KEY': Tensor([(n_frames), param_size])
                          Value=None if the param is added to dict as conditioning later
        """
        super().__init__()
        self.dag = dag
        self.name = name
        self.ext_param_sizes = {}
        self.ext_param_range = {}
        self.ext_param_types = {}
        self.processor_names = [processor.name for processor, connections in self.dag]
        self.fixed_param_names = list(fixed_params.keys())
        for k, v in fixed_params.items():
            if v is not None:
                self.register_buffer(k, v)
            else:
                setattr(self, k, None)
        self.processors = nn.ModuleList([]) # register modules for .to(device)
        for processor, connections in self.dag:
            self.processors.append(processor)
            # parameters that rely on external input and not outputs of other modules and are not fixed
            ext_params = [k for k, v in connections.items() if v not in self.processor_names+self.fixed_param_names]
            ext_sizes = {connections[k]: desc['size'] for k, desc in processor.get_param_desc().items() if k in ext_params}
            ext_range = {connections[k]: desc['range'] for k, desc in processor.get_param_desc().items() if k in ext_params}
            ext_types = {connections[k]: desc['type'] for k, desc in processor.get_param_desc().items() if k in ext_params}
            self.ext_param_sizes.update(ext_sizes)
            self.ext_param_range.update(ext_range)
            self.ext_param_types.update(ext_types)
            # {'ADD_AMP':1, 'ADD_HARMONIC': n_harmonics, 'CUTOFF': ...}
        self.ext_param_size = sum(self.ext_param_sizes.values())

    def fill_params(self, input_tensor, conditioning=None):
        """using network output tensor to fill synth parameter dict
        doesn't take into account parameter range

        Args:
            input_tensor (torch.Tensor): Shape [batch, n_frames, input_size]
                if parameters are stationary like a preset, n_frames should be 1
            conditioning: dict of conditions ex.) {'f0_hz': torch.Tensor [batch, n_frames_cond, 1]}
        Returns:
            dag_input: {'amp': torch.Tensor [batch, n_frames, 1], }
        """
        curr_idx = 0
        dag_input = {}
        batch_size = input_tensor.shape[0]
        n_frames = input_tensor.shape[1]
        device = input_tensor.device
        # parameters fed from input_tensor
        for ext_param, param_size in self.ext_param_sizes.items():
            scale_fn = SCALE_FNS[self.ext_param_types[ext_param]]
            split_input = input_tensor[:, :, curr_idx:curr_idx+param_size]
            p_range = self.ext_param_range[ext_param]
            dag_input[ext_param] = scale_fn(split_input, p_range[0], p_range[1])
            curr_idx += param_size
        # Fill fixed_params - no scaling applied
        for param_name in self.fixed_param_names:
            param_value = getattr(self, param_name)
            if param_value is None:
                value = conditioning[param_name] 
            elif len(param_value.shape) == 1:
                value = param_value[None, None, :].expand(batch_size, n_frames, -1).to(device)
            elif len(param_value.shape) == 2:
                value = param_value[None, :, :].expand(batch_size, -1, -1).to(device)
            dag_input[param_name] = value
        return dag_input

    def forward(self, dag_inputs, n_samples=None):
        """runs input through DAG of processors

        Args:
            dag_inputs (dict): ex. {'INPUT_KEY':Tensor}

        Returns:
            dict: Final output of processor
        """
        outputs = dag_inputs

        for node in self.dag:
            processor, connections = node
            inputs = {key: outputs[connections[key]] for key in connections}
            if n_samples and isinstance(processor, Gen):
                inputs.update({'n_samples': n_samples})
            # Run processor.
            signal = processor(**inputs)

            # Add the outputs of processor for use in subsequent processors
            outputs[processor.name] = signal # audio/control signal output

        #Use the output of final processor as signal
        output_name = self.dag[-1][0].name
        outputs[self.name] = outputs[output_name]
        
        return outputs[self.name], outputs
    
    def uniform(self, batch_size, n_samples, device):
        """
        assumes parameters requiring external input is stationary (n_frames=1)
        """
        n_frames = 1
        param_values = []
        for pn, psize in self.ext_param_sizes.items():
            prange = self.ext_param_range[pn]
            v = torch.rand(batch_size, n_frames, psize)*(prange[1] - prange[0]) + prange[0]
            param_values.append(v)
        param_tensor = torch.cat(param_values, dim=-1).to(device)
        dag_input = self.fill_params(param_tensor)
        audio, outputs = self(dag_input, n_samples)
        return param_tensor, audio