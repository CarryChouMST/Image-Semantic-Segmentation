import sys

sys.path.extend([r'D:\dl_project\dl_project_cnn\DataInput', r'D:\dl_project\dl_project_cnn\Models' \
                    , r'D:\dl_project\dl_project_cnn\Trainer', r'D:\dl_project\dl_project_cnn\Utils', '..'])
# print(sys.path)
import configparser

from DataInput.Dataset import BasicDataset
from torch.utils.data import DataLoader

from torch import optim
from Utils.lr_scheduler import *

from Utils.metrics import *
from Utils.loss import *
from Utils.visualizer import *
from Utils.saver import *

import logging

logging.getLogger().setLevel(logging.INFO)
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp

import os
import time


def train(net,
          config_path
          ):
    '''record the start time'''
    start_time = time.clock()

    '''read the config file'''
    cf = configparser.ConfigParser()
    cf.read(config_path)
    secs = cf.sections()
    batch_size = int(cf.get(secs[0], 'batch_size'))
    root_path = cf.get(secs[0], 'root_path')
    lr_base = float(cf.get(secs[0], 'lr_base'))
    epochs = int(cf.get(secs[0], 'epochs'))
    interval_step = int(cf.get(secs[0], 'interval_step'))
    val_step = int(cf.get(secs[0], 'val_step'))
    dir_checkpoint = cf.get(secs[0], 'dir_checkpoint')

    '''read input data, using BasicDataset and Dataloader'''
    train_dataset = BasicDataset(root=root_path, image_set='train', transform=False)
    val_dataset = BasicDataset(root=root_path, image_set='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)  # pred val_sample one by one
    max_iters = len(train_dataloader) * epochs

    '''appoint optimizer and lr_scheduler'''
    # weight_decay_list = (param for name, param in net.named_parameters() if 'bias' not in name)
    # no_decay_list = (param for name, param in net.named_parameters() if 'bias' in name)
    # parameters = [{'params': weight_decay_list},
    #               {'params': no_decay_list, 'weight_decay': 0.}]
    # optimizer = optim.SGD(parameters, lr=lr_base, momentum=0.9, nesterov=False, weight_decay=5e-4)
    optimizer = optim.AdamW(net.parameters(), lr=lr_base, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    lr_scheduler = ConsineAnnWithWarmup(optimizer=optimizer, loader_len=len(train_dataloader),  lr_max=lr_base, warm_prefix=True)
    # lr_scheduler = PolyLR(optimizer, base_lr=lr_base, max_iters=max_iters)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    '''set loss function and metrics class'''
    # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([15.84]).cuda())
    # criterion = nn.MSELoss()
    criterion = BCEDiceLoss(gamma=0.5)
    # l2_reg = l2_reg_loss(weight_decay=0.0005)
    train_metricer = StreamSegMetrics(n_classes=2)
    visualizer = Visualizer(comment=f'LR_{lr_base}_BS_{batch_size}_MAXiters_{max_iters}')

    '''monitoring iteration'''
    crt_iter = 0

    '''initialize the current best OA'''
    miou = 0
    saver = Saver(base_path=os.getcwd(), basis_name='MIOU_')

    '''initialize the net and optimizer'''
    net.cuda()
    use_amp = False
    # scaler = amp.GradScaler(enabled=use_amp)

    '''start training'''
    optimizer.zero_grad()
    for epoch in range(epochs):
        '''load net to cuda'''

        net.train()
        logging.info(f'start training {epoch+1}/{epochs}.....')

        '''through train_loader monitoring the progress in each epoch'''
        loader = tqdm(train_dataloader, desc=f'training {epoch+1}/{epochs}', unit='batch', total=len(train_dataloader))

        for batch in loader:

            '''update current iteration'''
            crt_iter += 1
            # lr_scheduler.update(c_iters=crt_iter)

            '''get data and load to cuda'''
            inputs = batch['image'].cuda()  # BCHW
            labels = batch['label'].cuda()

            '''reset the optimizer'''
            optimizer.zero_grad()

            '''predict the output'''
            # with amp.autocast(enabled=use_amp):
            #     outputs = net(inputs)
            #     # loss = criterion(outputs, labels) + l2_reg.forward(net)
            #     loss = criterion(outputs, labels)
            outputs = net(inputs)  # without sigmoid
            outputs_prob = nn.Sigmoid()(outputs.data)

            '''calculate the loss with l2'''
            loss = criterion(outputs, labels)

            '''back propagation, and update the parameters'''
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()

            # if crt_iter % (batch_size) == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.step()

            '''uodate the metric'''
            pred = (outputs_prob.data > 0.5).type(torch.float32)
            train_metricer.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())

            if crt_iter % interval_step == 0:
                '''monitoring on the cmd'''
                metric_dict = train_metricer.get_results()
                loader.set_description(f'Training {epoch+1}/{epochs} '
                                       f'Current loss:{"%.2f" % loss.data} '
                                       f'Precision:{"%.2f" % metric_dict["Precision_fore"]} '
                                       f'Recall:{"%.2f" % metric_dict["Recall_fore"]} '
                                       f'iou:{"%.2f" % metric_dict["Class IoU"][1]}')

                '''visualize, including lr, loss, acc'''
                visualizer.vis_scalar('lr', optimizer.param_groups[0]['lr'], crt_iter)
                visualizer.vis_scalar('train/loss', loss.data, crt_iter)
                visualizer.vis_scalar('train/Precision', metric_dict["Precision_fore"], crt_iter)
                visualizer.vis_scalar('train/Recall', metric_dict["Recall_fore"], crt_iter)
                visualizer.vis_scalar('train/iou', metric_dict["Class IoU"][1], crt_iter)

                '''transfer images to cpu'''
                inputs_cpu = inputs.data.cpu()
                label_cpu = labels.data.cpu()
                pred_cpu = pred.data.cpu()

                '''visualize'''
                visualizer.vis_images(tag='train_vis/image', img_tensor=inputs_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/label', img_tensor=label_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/pred', img_tensor=pred_cpu, global_step=crt_iter)

            if crt_iter % val_step == 0:

                '''validate the trained model'''
                loss_val, val_acc_dict = validate(net=net, val_loader=val_dataloader,
                                                  criterion=criterion, visualizer=visualizer, global_step=crt_iter)
                loader.set_description(f'validation {epoch+1}/{epochs} '
                                       f'loss:{"%.2f" % loss_val.data} '
                                       f'Precision:{"%.2f" % val_acc_dict["Precision_fore"]} '
                                       f'Recall:{"%.2f" % val_acc_dict["Recall_fore"]} '
                                       f'iou:{"%.2f" % val_acc_dict["Class IoU"][1]}')

                '''resave the current best model'''
                if val_acc_dict["Mean IoU"] >= miou:
                    saver.update(model=net, value=val_acc_dict["Mean IoU"])
                    miou = val_acc_dict["Mean IoU"]

                '''visualize, including lr, loss, acc'''
                visualizer.vis_scalar('val/loss', loss_val.data, crt_iter)
                visualizer.vis_scalar('val/Precision', val_acc_dict["Precision_fore"], crt_iter)
                visualizer.vis_scalar('val/Recall', val_acc_dict["Recall_fore"], crt_iter)
                visualizer.vis_scalar('val/iou', val_acc_dict["Class IoU"][1], crt_iter)

            lr_scheduler.step()

        '''after training a epoch completely, calculate the whole acc, save a checkpoint'''
        final_acc_dic = train_metricer.get_results()
        torch.save(net.state_dict(),
                   dir_checkpoint + f'CP_epoch{epoch + 1}MIOU{final_acc_dic["Mean IoU"]}.pth')  # 以字典形式保存了模型的所有参数
        logging.info(f'Checkpoint {epoch + 1} saved !')
        '''reset the train metric'''
        train_metricer.reset()

        '''complete a step (epoch) for lr_scheduler'''
        # lr_scheduler.step()

    '''complete all epochs, calculate and output running time'''
    end_time = time.clock()
    running_time = (end_time - start_time) / 3600
    logging.info(f'Running time: {running_time}hours')


def validate(net, val_loader, criterion, visualizer, global_step):
    '''change the network to validation mode'''
    net.eval()

    '''initialize the loss and metric'''
    val_metricer = StreamSegMetrics(n_classes=2)
    loss = 0

    '''create the iterater'''
    loader = tqdm(val_loader, desc=r'validating', unit='img', total=len(val_loader))
    visual_idx = np.random.randint(0, len(loader))

    for i, data in enumerate(loader):

        input = data['image'].cuda()
        label = data['label'].cuda()

        with torch.no_grad():
            output = net(input)
            output_prob = nn.Sigmoid()(output.data)
            pred = (output_prob.data > 0.5).type(torch.float32)

            '''update loss and metric'''
            loss = criterion(output, label) + loss
            val_metricer.update(label.data.cpu().numpy(), pred.data.cpu().numpy())

            if i == visual_idx:

                '''transfer images to cpu'''
                val_input_cpu = input.data.cpu()
                val_label_cpu = label.data.cpu()
                val_pred_cpu = pred.data.cpu()

                '''visualize'''
                visualizer.vis_images(tag='val_vis/image', img_tensor=val_input_cpu, global_step=global_step)
                visualizer.vis_images(tag='val_vis/label', img_tensor=val_label_cpu, global_step=global_step)
                visualizer.vis_images(tag='val_vis/pred', img_tensor=val_pred_cpu, global_step=global_step)

    res_dict = val_metricer.get_results()
    loss = loss / len(val_loader)

    '''change back to training'''
    net.train()

    return loss, res_dict



