"""Loading data - Modified for Pathology Graph Dataset"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json as jsonmod
import pandas as pd
import json
import nltk


class PathologyGraphDataset(data.Dataset):
    """
    Load pathology graph data
    Directory structure:
    graph/
        train/
            *.pt (graph files)
            train.txt (captions)
        dev/
            *.pt (graph files) 
            dev.txt (captions)
        test/
            *.pt (graph files)
            test.txt (captions)
    """

    def __init__(self, data_path, data_split, opt):
        self.data_path = data_path
        self.data_split = data_split
        
        # 构建图数据和caption的路径
        graph_dir = os.path.join(data_path, 'graph', data_split)
        caption_file = os.path.join(data_path, 'graph', data_split, f'{data_split}.txt')
        
        # 验证路径存在
        if not os.path.exists(graph_dir):
            raise ValueError(f"Graph directory not found: {graph_dir}")
        if not os.path.exists(caption_file):
            raise ValueError(f"Caption file not found: {caption_file}")
        
        # 加载caption - 保持原有格式
        self.captions = []
        with open(caption_file, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        
        # 加载图文件列表
        self.graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
        
        # 验证数据一致性
        if len(self.graph_files) != len(self.captions):
            print(f"Warning: Graph files ({len(self.graph_files)}) and captions ({len(self.captions)}) count mismatch")
            min_count = min(len(self.graph_files), len(self.captions))
            self.graph_files = self.graph_files[:min_count]
            self.captions = self.captions[:min_count]
        
        self.graph_dir = graph_dir
        self.length = len(self.graph_files)
        
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.graph_files) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
            
        # the development set for coco is large and so validation would be slow
       
            
        print(f'Pathology Graph Dataset - {data_split}: {self.length} graph-caption pairs')

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        if img_id >= len(self.graph_files):
            img_id = img_id % len(self.graph_files)
            
        # 加载图数据
        graph_file = self.graph_files[img_id]
        graph_path = os.path.join(self.graph_dir, graph_file)
        graph_data = torch.load(graph_path)
        
        # 获取caption - 保持原有处理方式
        caption = self.captions[index]
        
        # 转换为模型输入格式
        image, bbox = self._graph_to_input_format(graph_data)
        
        # Convert caption (string) to word ids - 保持原有处理方式
        caps = []
        caps.extend(caption.split(b','))
        caps = map(int, caps)

        captions = torch.Tensor(list(caps))
        
        return image, captions, bbox, index, img_id

    def _graph_to_input_format(self, graph_data):
        """Convert graph data to (image, bbox) format expected by the model"""
        # 节点特征作为图像区域特征
        node_features = graph_data.x
        num_nodes, feat_dim = node_features.shape
        
        # 特征维度处理到2048
        if feat_dim < 2048:
            padding = torch.zeros(num_nodes, 2048 - feat_dim)
            image = torch.cat([node_features, padding], dim=1)
        elif feat_dim > 2048:
            image = node_features[:, :2048]
        else:
            image = node_features
        
        # 节点数量处理到36
        if num_nodes < 36:
            padding = torch.zeros(36 - num_nodes, 2048)
            image = torch.cat([image, padding], dim=0)
        elif num_nodes > 36:
            image = image[:36]
        
        # 从节点坐标生成边界框
        centroids = graph_data.centroid
        bbox = self._coords_to_bbox(centroids, num_nodes)
        
        return image, bbox

    def _coords_to_bbox(self, centroids, num_nodes):
        """Convert node coordinates to bounding boxes"""
        bbox = torch.zeros(36, 4)
        
        if num_nodes > 0:
            # 归一化坐标到[0,1]范围
            min_coords, _ = centroids.min(dim=0)
            max_coords, _ = centroids.max(dim=0)
            range_coords = max_coords - min_coords
            range_coords[range_coords == 0] = 1.0
            
            normalized_coords = (centroids - min_coords) / range_coords
            
            # 生成边界框
            bbox_size = 0.02
            for i in range(min(num_nodes, 36)):
                x, y = normalized_coords[i][0].item(), normalized_coords[i][1].item()
                bbox[i] = torch.tensor([
                    max(0.0, x - bbox_size/2),
                    max(0.0, y - bbox_size/2),
                    min(1.0, x + bbox_size/2),
                    min(1.0, y + bbox_size/2)
                ])
        
        return bbox

    def __len__(self):
        return self.length


# 保持原有的PrecompDataset用于兼容性
class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split):
        # path
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc + '%s_bert_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)

        self.bbox = np.load(loc + '%s_ims_bbx.npy' % data_split)
        self.sizes = np.load(loc + '%s_ims_size.npy' % data_split, allow_pickle=True)

        print('image shape', self.images.shape)
        print('text shape', len(self.captions))

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index / self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]

        # Convert caption (string) to word ids.
        caps = []
        caps.extend(caption.split(','))
        caps = map(int, caps)

        # Load bbox and its size
        bboxes = self.bbox[img_id]
        imsize = self.sizes[img_id]
        # k sample
        k = image.shape[0]
        assert k == 36

        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize['image_w']
            bbox[1] /= imsize['image_h']
            bbox[2] /= imsize['image_w']
            bbox[3] /= imsize['image_h']
            bboxes[i] = bbox

        captions = torch.Tensor(caps)
        bboxes = torch.Tensor(bboxes)
        
        return image, captions, bboxes, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption) tuples.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, bboxes, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    bboxes = torch.stack(bboxes, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, bboxes, lengths, ids


def get_precomp_loader(data_path, data_split, opt, batch_size=64,
                       shuffle=True, num_workers=2):
    """Get loader for precomputed datasets"""
    dset = PrecompDataset(data_path, data_split)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_graph_loader(data_path, data_split, opt, batch_size=64,
                     shuffle=True, num_workers=2):
    """Get loader for pathology graph dataset"""
    dset = PathologyGraphDataset(data_path, data_split, opt)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    """Get train and validation loaders"""
    dpath = os.path.join(opt.data_path, data_name)
    
    # 自动检测数据类型
    graph_dir = os.path.join(dpath, 'graph')
    if os.path.exists(graph_dir):
        # 使用病理图数据集
        print(f"Using Pathology Graph Dataset from {graph_dir}")
        train_loader = get_graph_loader(dpath, 'train', opt, batch_size, True, workers)
        val_loader = get_graph_loader(dpath, 'dev', opt, batch_size, False, workers)
    else:
        # 使用原有的预计算数据集
        print(f"Using Precomputed Dataset from {dpath}")
        train_loader = get_precomp_loader(dpath, 'train', opt, batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', opt, batch_size, False, workers)
    
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt):
    """Get test loader"""
    dpath = os.path.join(opt.data_path, data_name)
    
    # 自动检测数据类型
    graph_dir = os.path.join(dpath, 'graph')
    if os.path.exists(graph_dir):
        # 使用病理图数据集
        test_loader = get_graph_loader(dpath, split_name, opt, batch_size, False, workers)
    else:
        # 使用原有的预计算数据集
        test_loader = get_precomp_loader(dpath, split_name, opt, batch_size, False, workers)
    
    return test_loader