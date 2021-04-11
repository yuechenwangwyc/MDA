import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
import random
from distutils.version import LooseVersion
import copy
from torch.autograd import Variable
import torch.nn.functional as F

MAIN_DIR = os.path.dirname(os.getcwd())


# --gpu_id 6  --dset office-home --s_dset_path Clipart --t_dset_path Art

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
    prep_dict["source1"] = prep.image_target(**config["prep"]['params'])
    prep_dict["source2"] = prep.image_target(**config["prep"]['params'])
    prep_dict["source3"] = prep.image_target(**config["prep"]['params'])
    prep_dict["target"] = prep.image_target(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source1"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source1"] = ImageList(open(data_config["source1"]["list_path"]).readlines(),
                                 transform=prep_dict["source1"])
    dset_loaders["source1"] = DataLoader(dsets["source1"], batch_size=train_bs,
                                         shuffle=True, num_workers=4, drop_last=True)

    dsets["source2"] = ImageList(open(data_config["source2"]["list_path"]).readlines(),
                                 transform=prep_dict["source2"])
    dset_loaders["source2"] = DataLoader(dsets["source2"], batch_size=train_bs,
                                         shuffle=True, num_workers=4, drop_last=True)

    dsets["source3"] = ImageList(open(data_config["source3"]["list_path"]).readlines(),
                                 transform=prep_dict["source3"])
    dset_loaders["source3"] = DataLoader(dsets["source3"], batch_size=train_bs,
                                         shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    ad_net1 = network.AdversarialNetwork(class_num, 1024)
    ad_net1 = ad_net1.cuda()
    ad_net2 = network.AdversarialNetwork(class_num, 1024)
    ad_net2 = ad_net2.cuda()
    ad_net3 = network.AdversarialNetwork(class_num, 1024)
    ad_net3 = ad_net3.cuda()

    ## set optimizer
    parameter_list = base_network.get_parameters() + ad_net1.get_parameters() + ad_net2.get_parameters() + ad_net3.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net1 = nn.DataParallel(ad_net1, device_ids=[int(i) for i, k in enumerate(gpus)])
        ad_net2 = nn.DataParallel(ad_net2, device_ids=[int(i) for i, k in enumerate(gpus)])
        ad_net3 = nn.DataParallel(ad_net3, device_ids=[int(i) for i, k in enumerate(gpus)])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i, k in enumerate(gpus)])

    ## train
    best_acc = 0.0
    iter_n = 0
    steps = 10
    threshold = 0.9
    for step in range(steps):
        print("#################### Part1 ####################")
        base_network.train(False)
        fin = open(data_config["target"]["list_path"])
        fout = open(config["pse_file"], "w")
        for i, (t_imgs, t_labels) in enumerate(dset_loaders["target"]):
            t_imgs = Variable(t_imgs).cuda()
            _, outputs = base_network(t_imgs)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs.data.cpu().numpy()
            ids = np.argmax(outputs, axis=1)
            for j in range(ids.shape[0]):
                line = fin.__next__()
                data = line.strip().split(" ")
                if outputs[j, ids[j]] >= threshold:
                    fout.write(data[0] + " " + str(ids[j]) + "\n")

        fin.close()
        fout.close()

        print("#################### Part2 ####################")
        for i in range(config["num_iterations"]):
            # test
            if i % config["test_interval"] == config["test_interval"] - 1:
                base_network.train(False)
                temp_acc = image_classification_test(dset_loaders, base_network)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_model = copy.deepcopy(base_network)
                log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
                config["out_file"].write(log_str + "\n")
                config["out_file"].flush()
                print(log_str)
            if i % 100 == 0:
                torch.save(best_model, config["out_model"])

            t_pse_set = ImageList(open(config["pse_file"]).readlines(),
                                  transform=prep_dict["target"])
            dset_loaders_pse = DataLoader(t_pse_set, batch_size=["source1"]["batch_size"],
                                          shuffle=True, num_workers=4)
            ## train one iter
            s1_loader, s2_loader, s3_loader = iter(dset_loaders["source1"]), iter(dset_loaders["source2"]), iter(
                dset_loaders["source3"])  # 78
            t_loader = iter(dset_loaders["target"])
            pse_loader = iter(dset_loaders_pse)

            base_network.train(True)
            ad_net1.train(True)
            ad_net2.train(True)
            ad_net3.train(True)
            for i, (inputs_source1, labels_source1) in enumerate(s1_loader):
                try:
                    inputs_source2, labels_source2 = s2_loader.next()
                except StopIteration:
                    s2_loader = iter(dset_loaders["source2"])
                    inputs_source2, labels_source2 = s2_loader.next()

                try:
                    inputs_source3, labels_source3 = s3_loader.next()
                except StopIteration:
                    s3_loader = iter(dset_loaders["source3"])
                    inputs_source3, labels_source3 = s3_loader.next()

                try:
                    inputs_target, _ = t_loader.next()
                except StopIteration:
                    t_loader = iter(dset_loaders["target"])
                    inputs_target, _ = t_loader.next()

                try:
                    inputs_pse, labels_pse = pse_loader.next()
                except StopIteration:
                    pse_loader = iter(dset_loaders_pse)
                    inputs_pse, labels_pse = pse_loader.next()

                loss_params = config["loss"]
                optimizer = lr_scheduler(optimizer, iter_n, **schedule_param)
                iter_n = iter_n + 1
                optimizer.zero_grad()
                # network
                inputs_source1, labels_source1 = inputs_source1.cuda(), labels_source1.cuda()
                inputs_source2, labels_source2 = inputs_source2.cuda(), labels_source2.cuda()
                inputs_source3, labels_source3 = inputs_source3.cuda(), labels_source3.cuda()
                inputs_pse, labels_pse = inputs_pse.cuda(), labels_pse.cuda()
                inputs_target = inputs_target.cuda()

                features_source1, outputs_source1 = base_network(inputs_source1)
                features_source2, outputs_source2 = base_network(inputs_source2)
                features_source3, outputs_source3 = base_network(inputs_source3)
                features_pse, outputs_pse = base_network(inputs_pse)
                features_target, outputs_target = base_network(inputs_target)

                outputs1 = torch.cat((outputs_source1, outputs_target), dim=0)
                softmax_out1 = nn.Softmax(dim=1)(outputs1)

                outputs2 = torch.cat((outputs_source2, outputs_target), dim=0)
                softmax_out2 = nn.Softmax(dim=1)(outputs2)

                outputs3 = torch.cat((outputs_source3, outputs_target), dim=0)
                softmax_out3 = nn.Softmax(dim=1)(outputs3)

                # loss calculation
                transfer_loss1 = loss.GVB(softmax_out1, ad_net1, network.calc_coeff(iter_n))
                transfer_loss2 = loss.GVB(softmax_out2, ad_net2, network.calc_coeff(iter_n))
                transfer_loss3 = loss.GVB(softmax_out3, ad_net3, network.calc_coeff(iter_n))
                classifier_loss1 = nn.CrossEntropyLoss()(outputs_source1, labels_source1)
                classifier_loss2 = nn.CrossEntropyLoss()(outputs_source2, labels_source2)
                classifier_loss3 = nn.CrossEntropyLoss()(outputs_source3, labels_source3)
                classifier_loss_pse = nn.CrossEntropyLoss()(outputs_pse, labels_pse)
                total_loss = classifier_loss1 + classifier_loss2 + classifier_loss3 + classifier_loss_pse + loss_params[
                    "trade_off"] * (
                                     transfer_loss1 + transfer_loss2 + transfer_loss3)

                if i % config["print_num"] == 0:
                    log_str = "iter: {:05d}, classifier_loss: {:.5f}".format(iter_n, total_loss)
                    config["out_file"].write(log_str + "\n")
                    config["out_file"].flush()
                    # print(log_str)

                total_loss.backward()
                optimizer.step()

    torch.save(best_model, config["out_model"])

    return best_acc


if __name__ == "__main__":

    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet50")
    parser.add_argument('--dset', type=str, default='office-home', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path1', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--s_dset_path2', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--s_dset_path3', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/Clipart.txt', help="The target dataset path list")  ###
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--print_num', type=int, default=100, help="interval of two print loss")
    parser.add_argument('--num_iterations', type=int, default=1000, help="interation num ")  ###
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")  ###
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for transfer loss")  ###
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
    config["log_dir"] = args.dset + "/log"
    config["pse"] = args.dset + "/pse"

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    if not osp.exists(config["log_dir"]):
        os.mkdir(config["log_dir"])

    config["out_file"] = open(
        osp.join(config["log_dir"], args.s_dset_path1 + "_" + args.s_dset_path2 + "_" + args.s_dset_path3 + "_" +
                 args.t_dset_path + "_" +
                 osp.abspath(__file__).split('/')[-1].split('.')[0] + '.txt'), "w")
    config["out_model"] = osp.join(config["output_path"],
                                   args.s_dset_path1 + "_" + args.s_dset_path2 + "_" + args.s_dset_path3 + "_" + args.t_dset_path + "_" +
                                   osp.abspath(__file__).split('/')[-1].split('.')[0] + '.pth')
    config["pse_file"] = os.path.join(config["pse"],
                                      args.s_dset_path1 + "_" + args.s_dset_path2 + "_" + args.s_dset_path3 + "_" +
                                      args.t_dset_path + "_" +
                                      osp.abspath(__file__).split('/')[-1].split('.')[0] + '.txt')

    config["prep"] = {'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": args.trade_off}
    if "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": False, "bottleneck_dim": 256,
                                        "new_cls": True}}
    else:
        raise ValueError('Network cannot be recognized. Please define your own dataset here.')

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source1": {"list_path": osp.join(MAIN_DIR, "dataset", args.dset, args.s_dset_path1 + ".txt"),
                                  "batch_size": args.batch_size},
                      "source2": {"list_path": osp.join(MAIN_DIR, "dataset", args.dset, args.s_dset_path2 + ".txt"),
                                  "batch_size": args.batch_size},
                      "source3": {"list_path": osp.join(MAIN_DIR, "dataset", args.dset, args.s_dset_path3 + ".txt"),
                                  "batch_size": args.batch_size},
                      "target": {"list_path": osp.join(MAIN_DIR, "dataset", args.dset, args.t_dset_path + ".txt"),
                                 "batch_size": args.batch_size},
                      "test": {"list_path": osp.join(MAIN_DIR, "dataset", args.dset, args.t_dset_path + ".txt"),
                               "batch_size": args.batch_size}}

    if config["dataset"] == "office-home":
        seed = 2019
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "office":
        seed = 2019
        if ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "visda":
        seed = 9297
        config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    # print(seed)
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