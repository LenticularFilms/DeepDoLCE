from models import uresnet

def get_model(args,data_stats):

    #### setup network ############################################################################
    if args.model == 'UResNet':
        return uresnet.UResNet(args,data_stats)
    else:
        raise NotImplementedError
