"""Model utilities for SteptronOss."""


def init_weight_callback(cls, init_method="init_model_weight"):
    """this func is called after the model's __init__ method is executed"""

    def _recursive_init_model_weights(model, init_method):
        for child in model.children():
            _recursive_init_model_weights(child, init_method)
        if hasattr(model, init_method):
            getattr(model, init_method)()

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Only execute the callback if the actual class of the instance is the class to which the decorator is applied
        if self.__class__ is cls:
            _recursive_init_model_weights(self, init_method)

    cls.__init__ = new_init
    return cls
