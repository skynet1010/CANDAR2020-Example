from torch import nn
from code.utils.consts import nr_of_classes, model_dict


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def manipulateModel(model_name, is_feature_extraction,dim):
    model = model_dict[model_name](pretrained=is_feature_extraction)
    set_parameter_requires_grad(model, True)
    #output layer
    if model_name == "resnet18" or \
        model_name == "wide_resnet50_2":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nr_of_classes)
    elif model_name =="alexnet":
        layer_number = 6
        num_ftrs = model.classifier[layer_number].in_features
        model.classifier[layer_number] = nn.Linear(num_ftrs,nr_of_classes)
 
    #input layer:
    if dim!=3:
        if model_name =="alexnet":
            layer_index = 0
            model.features[layer_index] = nn.Conv2d(dim,64,kernel_size=(11,11),stride=(4,4),padding=(2,2))
        elif model_name == "resnet18" or model_name == "wide_resnet50_2":
            model.conv1 = nn.Conv2d(dim,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

    return model.cuda()
