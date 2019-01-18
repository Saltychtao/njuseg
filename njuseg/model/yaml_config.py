import yaml

def parse_options(parser):
    """Parse options from command line arguments and optionally config file
    :return:
    Options
    argparse.Namespace
    """
    opt = parser.parse_args()
    if opt.config_file:
        with open(opt.config_file,'r') as f:
            data = yaml.load(f)
            arg_dict = opt.__dict__
            for key,value in data.items():
                if isinstance(value,dict):
                    if key == opt.env:
                        for k,v in value.items():
                            arg_dict[k] = v
                    else:
                        continue
                else:
                    arg_dict[key] = value
    return opt

def print_config(opt,**kwargs):
    print('===== CONFIG =====')
    for k,v in vars(opt).items():
        print (k,':',v)
    for k,v in kwargs.items():
        print(k,':',v)
    print('==================')
