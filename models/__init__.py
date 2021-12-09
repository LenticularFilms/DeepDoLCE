from . import uresnet,uresnet_colorize

def get_model(args,data_stats,secondary_model=False):

    model_name = args.model
    if secondary_model == True:
        model_name = args.model_2
    if model_name == 'UResNet':
        return uresnet.UResNet(args,data_stats)
    if model_name == 'UResNetColorize':
        return uresnet_colorize.Adversarialmodel(args,data_stats)
    else:
        raise NotImplementedError
