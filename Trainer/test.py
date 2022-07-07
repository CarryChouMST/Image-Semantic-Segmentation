import sys
sys.path.extend([r'D:\dl_project\dl_project_cnn\DataInput', r'D:\dl_project\dl_project_cnn\Models'\
                    , r'D:\dl_project\dl_project_cnn\Trainer', r'D:\dl_project\dl_project_cnn\Utils'])

import configparser

from DataInput.Dataset import BasicDataset
from torch.utils.data import DataLoader

from Utils.metrics import *
from Utils.loss import *
from Utils.visualizer import *

from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np


def test(net,
          config_path
          ):

    '''read the config file'''
    cf = configparser.ConfigParser()
    cf.read(config_path)
    secs = cf.sections()
    root_path = cf.get(secs[0], 'root_path')
    test_model_path = cf.get(secs[0], 'test_model_path')

    '''read input data, using BasicDataset and Dataloader'''
    test_dataset = BasicDataset(root=root_path, image_set='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)  # pred test_sample one by one

    '''set loss function and metrics class'''
    criterion = nn.BCEWithLogitsLoss()
    test_metricer = StreamSegMetrics(n_classes=2)
    visualizer = Visualizer(comment='test')

    '''load net to cuda, load the params'''
    net.load_state_dict(torch.load(test_model_path))
    net.cuda()
    net.eval()

    '''initialize the loss value'''
    loss = 0

    '''through test_loader monitoring the progress in each epoch'''
    loader = tqdm(test_dataloader, desc=f'testing', unit='img', total=len(test_dataloader))
    for i, data in enumerate(loader):

        input = data['image'].cuda()
        label = data['label'].cuda()

        with torch.no_grad():
            output = net(input)
            pred = (output > 0.5).type(torch.float32)  # BCHW Tensor in GPU

            '''update loss and metric'''
            loss = criterion(output, label) + loss
            test_metricer.update(label.data.cpu().numpy(), pred.data.cpu().numpy())

        if i == 0:

            label_vis = label[0].data.repeat(3, 1, 1)
            pred_vis = pred[0].data.repeat(3, 1, 1)
            concat_img = torch.stack((input[0].data, label_vis, pred_vis), dim=0)
            visualizer.vis_images(tag='train_vis', img_tensor=concat_img, global_step=i)


    res_dict = test_metricer.get_results()
    loss = loss/len(loader)

    return loss, res_dict

def dilated_test(net, config_path, size=300, dilated_size=512, gamma=0.5, update=False):
    
    '''read the config file'''
    cf = configparser.ConfigParser()
    cf.read(config_path)
    secs = cf.sections()
    root_path = cf.get(secs[0], 'root_path')
    test_model_path = cf.get(secs[0], 'test_model_path')

    '''read input data, using BasicDataset and Dataloader'''
    test_dataset = BasicDataset(root=root_path, image_set='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)  # pred test_sample one by one

    '''set loss function and metrics class'''
    criterion = nn.BCEWithLogitsLoss(torch.tensor([16.0]).cuda())
    # criterion = nn.MSELoss()
    # criterion = BCEDiceLoss()
    test_metricer = StreamSegMetrics(n_classes=2, buffer_size=1)
    visualizer = Visualizer(comment='test')

    '''load net to cuda, load the params'''
    if update:
        net_dict = net.state_dict()
        weight_dict = torch.load(test_model_path)
        weight_dict = {k:v for k, v in weight_dict.items() if k in net_dict}
        net_dict.update(weight_dict)
        net.load_state_dict(net_dict)
        logging.info(f'loaded!')
    else:
        net.load_state_dict(torch.load(test_model_path))
    logging.info(f'loading {test_model_path} finished!')
    net.cuda()
    net.eval()

    '''initialize the loss value'''
    loss = 0

    '''through test_loader monitoring the progress in each epoch'''
    loader = tqdm(test_dataloader, desc=f'testing', unit='img', total=len(test_dataloader))
    visual_list = np.random.randint(0, len(loader), 40)

    for i, data in enumerate(loader):
        # if i < 1:

        input_raw = data['image'].detach()  # BCHW 1500*1500
        label_raw = data['label'].detach()

        '''crop to inference'''
        use_to_inference_map = get_dilated_inference_map(img=input_raw, size=size, input_size=dilated_size)
        input_list = img_patchs(map=use_to_inference_map, img=input_raw, input_size=dilated_size)  # NBCHW
        label_list = img_patchs(map=use_to_inference_map, img=label_raw, input_size=dilated_size)  # NBCHW

        '''inference each input'''
        results=[]
        temp_loss = 0

        for i_i, (input, label) in enumerate(zip(input_list, label_list)):
            with torch.no_grad():
                input = input.cuda()
                label = label.cuda()
                output = net(input.cuda())  # before sigmoid
                output_prob = torch.sigmoid(output.data)
                pred = (output_prob > gamma).type(torch.float32).data.cpu()  # BCHW Tensor in CPU
                results.append(pred)
                temp_loss = criterion(output.data, label.data) + temp_loss

        pred_raw = pred_fusion(map=use_to_inference_map, results=results, img_shape=label_raw.shape, size=size)  # BCHW  CPU

        '''update loss and metric'''
        loss += temp_loss/(i_i+1)
        test_metricer.update(label_raw.cpu().numpy(), pred_raw.cpu().numpy())

        if i in visual_list:
            visualizer.vis_images(tag='test_vis/image', img_tensor=input_raw, global_step=i+1)
            visualizer.vis_images(tag='train_vis/label', img_tensor=label_raw, global_step=i+1)
            visualizer.vis_images(tag='train_vis/pred', img_tensor=pred_raw, global_step=i+1)


    res_dict = test_metricer.get_results()
    loss = loss/len(loader)

    return loss, res_dict

def get_dilated_inference_map(img, size, input_size): 
    use_to_inference_map = {}  # {pred start point:crop to inference start point, usable result start point}
    img = img.squeeze(0).numpy().transpose((1, 2, 0)) # img:numpy HWC 1500*1500
    height = img.shape[0]
    weight = img.shape[1]
    x_point_index = [i for i in range(0, weight, size)]
    y_point_index = [i for i in range(0, height, size)]
    for x_point in x_point_index:
        for y_point in y_point_index:
            
            if x_point == 0 and y_point == 0:
                use_to_inference_map[(x_point, y_point)] = [(0,0), (0,0)]
                
            elif x_point == 0 and (y_point not in (0, height-size)):
                use_to_inference_map[(x_point, y_point)] = [(0, y_point-(input_size-size)//2),(0, (input_size-size)//2)]
                
            elif x_point == 0 and y_point == height-size:
                use_to_inference_map[(x_point, y_point)] = [(0, y_point-(input_size-size)),(0, (input_size-size))]
                
            elif x_point == weight-size and y_point == 0:
                use_to_inference_map[(x_point, y_point)] = [(x_point-(input_size-size),0), (input_size-size,0)]
                
            elif x_point == weight-size and (y_point not in (0, 1500-size)):
                use_to_inference_map[(x_point, y_point)] = \
                    [(x_point-(input_size-size),y_point-(input_size-size)//2), (input_size-size,(input_size-size)//2)]
                    
            elif x_point == weight-size and y_point == height-size:
                use_to_inference_map[(x_point, y_point)] = \
                    [(x_point-(input_size-size),y_point-(input_size-size)), (input_size-size,input_size-size)]
                    
            elif y_point == 0 and (x_point not in (0, weight-size)):
                use_to_inference_map[(x_point, y_point)] =[(x_point-(input_size-size)//2, 0), ((input_size-size)//2, 0)]
                
            elif y_point == height-size and (x_point not in (0, weight-size)):
                use_to_inference_map[(x_point, y_point)] = \
                    [(x_point-(input_size-size)//2, y_point-(input_size-size)), ((input_size-size)//2, input_size-size)]
                    
            else:
                use_to_inference_map[(x_point, y_point)] = \
                    [(x_point-(input_size-size)//2,y_point-(input_size-size)//2), ((input_size-size)//2,(input_size-size)//2)]
                    
    return use_to_inference_map

def img_patchs(map, img, input_size=512):  # img BCHW Tensor
    img_list = [None]*len(map)

    for i, key in enumerate(map):
        temp_x, temp_y = map[key][0]
        temp_to_inference = img[:, :, temp_y:temp_y+input_size, temp_x:temp_x+input_size] # BCHW
        img_list[i] = temp_to_inference
    return img_list

def pred_fusion(map, img_shape, results, size=300):
    img = torch.Tensor(img_shape)  #CPU_Tensor_BCHW
    for i, key in enumerate(map):
        raw_x, raw_y = key
        index_x, index_y = map[key][1]
        img[:, :, raw_y:raw_y+size, raw_x:raw_x+size] = results[i][:, :, index_y:index_y+size, index_x:index_x+size]
    return img

        
        
                
                
    


