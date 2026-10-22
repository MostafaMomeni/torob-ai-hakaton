from importlib import import_module
mod = import_module('orchestrator')
for k, v in mod.__dict__.items():
    globals()[k] = v
