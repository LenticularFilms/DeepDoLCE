import os
import io

def new_log(base_path,base_name,args=None):
    name = base_name

    folder_path = os.path.join(base_path,args.data) 
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    folder_path = os.path.join(folder_path,base_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    previous_runs = os.listdir(folder_path)
    n_exp = len(previous_runs)

    experiment_folder = os.path.join(folder_path,"experiment_{}".format(n_exp))

    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        with open(os.path.join(experiment_folder, "args" + '.txt'), 'w') as f:
            sorted_names = sorted(args_dict.keys(), key=lambda x: x.lower())
            for key in sorted_names:
                value = args_dict[key]
                f.write('%s:%s\n' % (key, value))

    return experiment_folder