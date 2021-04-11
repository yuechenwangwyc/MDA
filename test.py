import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
import random
from distutils.version import LooseVersion

MAIN_DIR=os.path.dirname(os.getcwd())
#--dset office-home --s_dset_path Clipart --t_dset_path Art --gpu_id 6

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
       iter_test = iter(loader["test"])
       for i in range(len(loader['test'])):
           data = iter_test.next()
           inputs = data[0]
           labels = data[1]
           inputs = inputs.cuda()
           labels = labels.cuda()
           _, outputs = model(inputs)
           if start_test:
               all_output = outputs.float()
               all_label = labels.float()
               start_test = False
           else:
               all_output = torch.cat((all_output, outputs.float()), 0)
               all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_dict["source"] = prep.image_target(**config["prep"]['params'])
    prep_dict["target"] = prep.image_target(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    #base_network=torch.load('/data/wyc/MDA/office-home/snapshot/Clipart_Product_Real_World_Art_baseline_n_adv.pth')

    base_network = base_network.cuda()

    ## add additional network for some methods

 
    ## set optimizer
    # parameter_list = base_network.get_parameters()
    # optimizer_config = config["optimizer"]
    # optimizer = optimizer_config["type"](parameter_list, \
    #                 **(optimizer_config["optim_params"]))
    # param_lr = []
    # for param_group in optimizer.param_groups:
    #     param_lr.append(param_group["lr"])
    # schedule_param = optimizer_config["lr_param"]
    # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    #multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i,k in enumerate(gpus)])

    base_network = torch.load('/data/wyc/Clipart_Product_Real_World_Art_baseline_n_adv.pth')
    
    ## train
    best_acc = 0.0
    iter_n=0
    base_network.eval()
    temp_acc = image_classification_test(dset_loaders, base_network)
    temp_model = nn.Sequential(base_network)
    if temp_acc > best_acc:
        best_acc = temp_acc
        best_model = temp_model
    log_str = "iter: {:05d}, precision: {:.5f}".format(0, temp_acc)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    print(log_str)


    return best_acc

if __name__ == "__main__":

    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet50")
    parser.add_argument('--dset', type=str, default='office-home', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/Clipart.txt', help="The target dataset path list")###
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--print_num', type=int, default=100, help="interval of two print loss")
    parser.add_argument('--num_iterations', type=int, default=1000, help="interation num ")###
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")###
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for transfer loss")###
    parser.add_argument('--batch_size', type=int, default=36, help="batch size")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations 
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["output_for_test"] = True
    config["output_path"] = args.dset + "/snapshot"
    config["log_dir"]= args.dset + "/log"

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    if not osp.exists(config["log_dir"]):
        os.mkdir(config["log_dir"])

    config["out_file"] = open(osp.join(config["log_dir"], args.s_dset_path + "_" +
                                       args.t_dset_path + "_" +
                                       osp.abspath(__file__).split('/')[-1].split('.')[0] + '.txt'), "w")
    config["out_model"] = osp.join(config["output_path"], args.s_dset_path + "_" + args.t_dset_path + "_" +
                                   osp.abspath(__file__).split('/')[-1].split('.')[0] + '.pth')


    config["prep"] = {'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off}
    if "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":False, "bottleneck_dim":256, "new_cls":True} }
    else:
        raise ValueError('Network cannot be recognized. Please define your own dataset here.')

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":osp.join(MAIN_DIR,"dataset",args.dset,args.s_dset_path+".txt"), "batch_size":args.batch_size}, \
                      "target":{"list_path":osp.join(MAIN_DIR,"dataset",args.dset,args.t_dset_path+".txt"), "batch_size":args.batch_size}, \
                      "test":{"list_path":osp.join(MAIN_DIR,"dataset",args.dset,args.t_dset_path+".txt"), "batch_size":args.batch_size}}

    if config["dataset"] == "office-home":
        seed = 2019
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "office":
        seed = 2019
        if   ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "visda":
        seed = 9297
        config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    #print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
