import torch
import torch.nn as nn
from diffsynth.processor import Gen


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
        self.processor_names = [processor.name for processor, connections in self.dag]
        self.fixed_param_names = list(fixed_params.keys())
        self.fixed_params = fixed_params
        for processor, connections in self.dag:
            # parameters that rely on external input and not outputs of other modules and are not fixed
            ext_params = [k for k, v in connections.items() if v not in self.processor_names+self.fixed_param_names]
            ext_sizes = {connections[k]:v for k,v in processor.get_param_sizes().items() if k in ext_params}
            ext_range = {connections[k]:v for k,v in processor.get_param_range().items() if k in ext_params}
            self.ext_param_sizes.update(ext_sizes)
            self.ext_param_range.update(ext_range)
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
        # parameters fed from input_tensor
        for ext_param, param_size in self.ext_param_sizes.items():
            dag_input[ext_param] = input_tensor[:,:, curr_idx:curr_idx+param_size]
            curr_idx += param_size
        for param_name, param_value in self.fixed_params.items():
            if param_value is None:
                value = conditioning[param_name] 
            elif len(param_value.shape) == 1:
                value = param_value[None, None, :].expand(batch_size, n_frames, -1)
            elif len(param_value.shape) == 2:
                value = param_value[None, :, :].expand(batch_size, -1, -1)
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