"""VGG-6 network and optimizer factories matching the supplied notebook."""
import logging
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("vgg6.model")

ACTIVATIONS = ("relu", "sigmoid", "tanh", "silu", "gelu")
OPTIMIZERS = ("sgd", "nesterov-sgd", "adam", "adagrad", "rmsprop", "nadam")

def make_activation(name: str) -> nn.Module:
    factories = {
        "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
        "silu": nn.SiLU, "gelu": nn.GELU,
    }
    key = name.lower()
    if key not in factories:
        raise ValueError(f"Unsupported activation {name!r}; choose from {ACTIVATIONS}")
    logger.debug("ACTIVATION CREATED | name=%s", key)
    return factories[key]()

class VGG6(nn.Module):
    """Four convolution layers, two pooling layers, global average pooling, classifier."""
    def __init__(self, activation: str = "relu", num_classes: int = 10):
        super().__init__()
        self.activation_name = activation.lower()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), make_activation(activation),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), make_activation(activation),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), make_activation(activation),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), make_activation(activation),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        self._initialize_weights()
        logger.info("MODEL CREATED | architecture=VGG6 activation=%s parameters=%d classes=%d", self.activation_name, sum(p.numel() for p in self.parameters()), num_classes)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.activation_name in ("sigmoid", "tanh"):
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))

def make_optimizer(name: str, model, learning_rate: float):
    key = name.lower()
    logger.info("OPTIMIZER CREATED | name=%s learning_rate=%s", key, learning_rate)
    if key == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    if key == "nesterov-sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    if key == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    if key == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    if key == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    if key == "nadam":
        return optim.NAdam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer {name!r}; choose from {OPTIMIZERS}")
