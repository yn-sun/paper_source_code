import random
import torch.nn as nn

class NASearchSpace:
    def __init__(self, input_channels=3, num_classes=10, max_layers=8):
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.kernel_sizes = [3, 5, 7]
        self.expansion_ratios = [1, 2, 4]
        self.depth_options = list(range(1, max_layers+1))

    def sample_arch(self):
        num_layers = random.choice(self.depth_options)
        layers = []
        for _ in range(num_layers):
            layers.append({
                'kernel': random.choice(self.kernel_sizes),
                'expansion': random.choice(self.expansion_ratios)
            })
        return {'num_layers': num_layers, 'layers': layers}

    def encode(self, arch):
        code = [self.depth_options.index(arch['num_layers'])]
        for layer in arch['layers']:
            code.append(self.kernel_sizes.index(layer['kernel']))
            code.append(self.expansion_ratios.index(layer['expansion']))
        return code

    def decode(self, code):
        num_layers = self.depth_options[code[0]]
        layers = []
        idx = 1
        for _ in range(num_layers):
            k = self.kernel_sizes[code[idx]]
            e = self.expansion_ratios[code[idx+1]]
            layers.append({'kernel': k, 'expansion': e})
            idx += 2
        return {'num_layers': num_layers, 'layers': layers}

    def random_population(self, N):
        return [self.encode(self.sample_arch()) for _ in range(N)]

    def build_model(self, code):
        cfg = self.decode(code)
        layers = []
        in_c = self.input_channels
        for layer_cfg in cfg['layers']:
            out_c = in_c * layer_cfg['expansion']
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=layer_cfg['kernel'], padding=layer_cfg['kernel']//2))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_c, self.num_classes))
        return nn.Sequential(*layers)
