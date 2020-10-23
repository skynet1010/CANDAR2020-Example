import torchvision.models as models
import torch



data_compositions = {
    "RGB_SLOPE":4,
    "NDVI_SLOPE":2,
    "RGB_NIR_SLOPE":5
}

model2data_compositions = {
    "alexnet":"RGB_SLOPE",
    "wide_resnet50_2":"NDVI_SLOPE",
    "resnet18":"RGB_NIR_SLOPE"
}

model_dict = \
{
    "resnet18":models.resnet18,
    "alexnet":models.alexnet,
    "wide_resnet50_2":models.wide_resnet50_2
}

nr_of_classes = 2

loss_dict = {
    "BCELoss":torch.nn.BCELoss,
    "MSELoss":torch.nn.MSELoss
}

model2loss_dict={
    "alexnet":"MSELoss",
    "wide_resnet50_2":"BCELoss",
    "resnet18":"BCELoss"
}