# ==============================================================================
# [Custom Designed for KWS20 Optimization - Final Selected Architecture]
# Note: This ResNet-lite architecture mitigates information loss and accuracy degradation 
# caused by 8-bit Quantization-Aware Training (QAT). It utilizes ai8x.Add() for 
# hardware-accelerated residual shortcuts.
# Features Dynamic Depth (via EXTRA_DEPTH env variable) for Latency-Accuracy trade-off tests.
# ==============================================================================

import os
from torch import nn
import ai8x

class AI87KWS20Net_ResNet(nn.Module):
    def __init__(
            self,
            num_classes=21,
            num_channels=128, 
            base_width=int(os.environ.get('BASE_WIDTH', 128)),            
            dimensions=(128, 1),
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        
        # [Optimization] Dynamic depth scaling for latency evaluation
        self.extra_depth = int(os.environ.get('EXTRA_DEPTH', 0))
        
        w1 = int(base_width * 1.5)  
        w2 = int(base_width * 1.5)  
        w3 = base_width             
        w4 = int(base_width * 0.75) 

        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, w1, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.voice_conv2 = ai8x.FusedConv1dReLU(w1, w2, 3, stride=1, padding=0, bias=bias, **kwargs)
        
        # Additional deep layers if EXTRA_DEPTH > 0
        self.extra_convs = nn.ModuleList()
        self.extra_adds = nn.ModuleList()
        for _ in range(self.extra_depth):
            self.extra_convs.append(ai8x.Conv1d(w2, w2, 3, stride=1, padding=1, bias=bias, **kwargs))
            self.extra_adds.append(ai8x.Add())

        self.voice_conv3 = ai8x.FusedMaxPoolConv1d(w2, w3, 3, stride=1, padding=0, bias=bias, **kwargs)        
        
        # [Architecture] Shortcut added to preserve identity features through QAT clipping
        self.shortcut = ai8x.FusedMaxPoolConv1d(w2, w3, 3, stride=1, padding=0, bias=bias, **kwargs) 
        
        self.voice_conv4 = ai8x.FusedConv1dReLU(w3, w4, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(w4, w3, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv2 = ai8x.FusedConv1dReLU(w3, w2, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(w2, w1, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(w1, w3, 3, stride=1, padding=0, bias=bias, **kwargs)
        
        fc_input_dim = w3 * 3 
        self.fc = ai8x.Linear(fc_input_dim, num_classes, bias=bias, wide=True, **kwargs)
        
        # Hardware-aware addition layer
        self.residual_add = ai8x.Add()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        
        # Dynamic depth routing
        for i in range(self.extra_depth):
            res = x
            x = self.extra_convs[i](x)
            x = self.extra_adds[i](x, res)
            x = self.relu(x)

        identity = x  
        x = self.voice_conv3(x)  
        identity = self.shortcut(identity)  
        
        x = self.residual_add(x, identity)
        x = self.relu(x)        
        
        x = self.voice_conv4(x)
        x = self.drop(x)
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        x = self.drop(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ai87kws20net_resnet(pretrained=False, **kwargs):
    assert not pretrained
    return AI87KWS20Net_ResNet(**kwargs)

models = [
    {
        'name': 'ai87kws20net_resnet',
        'min_input': 1,
        'dim': 1,
    },
]