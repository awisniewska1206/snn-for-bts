"""Helpers for normalizing trainer model config access."""


def canonicalize_model_type(model_type):
    """Normalize legacy model aliases to the runtime model names."""
    if model_type is None:
        raise ValueError("model_type is required")

    normalized = str(model_type).strip().lower()
    aliases = {
        "rnn": "gru",
    }
    return aliases.get(normalized, normalized)


def ensure_model_config_schema(model_cfg):
    """
    Fill a few safe defaults so trainer code can read directly from config.

    The trainer now expects the recurrent/MLP/TCN configs to expose fields like
    ``model.hidden_size`` and ``model.dropout`` directly.
    """
    if model_cfg is None:
        raise ValueError("model config is required")

    if "trainable" not in model_cfg:
        model_cfg["trainable"] = True

    if "input_network" not in model_cfg or model_cfg["input_network"] is None:
        model_cfg["input_network"] = {}

    input_network_cfg = model_cfg["input_network"]
    if "input_trainable" not in input_network_cfg:
        input_network_cfg["input_trainable"] = True
    if "input_layer_dropout" not in input_network_cfg:
        input_network_cfg["input_layer_dropout"] = 0.0

    return model_cfg
