# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:31:09 2021

@author: pedro
"""

import torch

torch.cuda.is_available()

torch.cuda.current_device()

torch.cuda.device(0)

torch.cuda.device_count()

torch.cuda.get_device_name(0)