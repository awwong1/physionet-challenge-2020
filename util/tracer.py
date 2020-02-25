"""Utility functions for iterating through the nested PyTorch modules.
"""
from collections import namedtuple
from pprint import pformat

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def walk_modules(module, name=None, path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if name is None:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class ActivationTracer(object):
    """Layer by layer intermediary forward activations hook manager.
    """

    def __init__(self, model, enabled=True, paths=None):
        self._model = model
        self.enabled = enabled
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self.activations = {}

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("ActivationTracer is not reentrant")
        self.entered = True
        self._forwards = {}  # store original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, walk_modules(self._model)))
        del self._forwards
        self.exited = True

    def __str__(self):
        if self.exited:
            return pformat(
                dict((k, v[2].shape) for (k, v) in self.activations.items())
            )
        return "<unfinished ActivationTracer>"

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        (path, leaf, module) = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            remove_handle = module.register_forward_hook(self._hook_fn_factory(trace))
            if path in self._forwards:
                raise RuntimeError(
                    "ActivationTracer path collision for {}".format(path)
                )
            self._forwards[path] = remove_handle

    def _hook_fn_factory(self, trace):
        def _hook_fn(module, inp, output):
            self.activations[trace.path] = (module, inp, output)

        return _hook_fn

    def _remove_hook_trace(self, trace):
        remove_handle = self._forwards.get(trace.path, None)
        if remove_handle is not None:
            remove_handle.remove()

    def items(self):
        return self.activations.items()
