from . import uresnet,ganv1,uresnet_colorize

def get_model(args,data_stats,secondary_model=False):

    #### setup network ############################################################################
    model_name = args.model
    if secondary_model == True:
        model_name = args.model_2

    if model_name == 'UResNet':
        return uresnet.UResNet(args,data_stats)
    if model_name == 'GANv1':
        return ganv1.GANmodel(args,data_stats)
    if model_name == 'UResNetColorize':
        #return uresnet_colorize.UResNet(args,data_stats)
        return uresnet_colorize.Adversarialmodel(args,data_stats)
    else:
        raise NotImplementedError
