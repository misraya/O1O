# ------------------------------------------------------------------------
# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out
# Misra Yavuz, Fatma Guney
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection 
# Orr Zohar, Jackson Wang, Serena Yeung
# -----------------------------------------------------------------------
# partly taken from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py

import functools
import torch
import os
import tarfile
import collections
import copy
from torchvision.datasets import VisionDataset
import itertools
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url
from torchvision.ops import box_iou, batched_nms
from datasets.coco_api import COCO

VOC_COCO_CLASS_NAMES={}
UNK_CLASS = ["unknown"]

# SOWOD splits
T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball", "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "banana","apple","sandwich", "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","tvmonitor",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","bottle"
]

sowodb_t4_superclasses = {
    "animal": ["bear", "bird", "cat", "cow", "dog", "elephant", "giraffe", "horse", "sheep", "zebra"],
    "vehicle": ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "truck"],
    "person": ["person"],
    "accessories": ["backpack","umbrella","handbag","tie","suitcase"],
    "appliances": ["microwave","oven","toaster","sink","refrigerator"],
    "furniture": ["chair", "diningtable", "pottedplant", "sofa", "bed", "toilet"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "food": ["banana","apple","sandwich", "orange","broccoli","carrot","hot dog","pizza","donut","cake"],
    "sports": ["frisbee","skis","snowboard","sports ball", "kite","baseball bat","baseball glove","skateboard", "surfboard","tennis racket"],
    "electronic": ["laptop","mouse","remote","keyboard","cell phone", "tvmonitor"],
    "indoor": ["book","clock","vase","scissors","teddy bear","hair drier","toothbrush"],
    "kitchen": ["wine glass","cup","fork","knife","spoon","bowl","bottle"],
    "unknown": ["unknown"]
}

VOC_COCO_CLASS_NAMES["OWDETR"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

# MOWOD splits
VOC_CLASS_NAMES = [
"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

T2_CLASS_NAMES = [
    "elephant", "bear", "zebra", "giraffe",
    "truck", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", 
    "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

mowodb_t4_superclasses = {
    "animal": ["bird", "cat", "cow", "dog", "horse", "sheep", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "truck"],
    "person": ["person"],
    "furniture": ["chair", "diningtable", "pottedplant", "sofa", "bed", "toilet"],
    "electronic": ["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
    "unknown": ["unknown"]
}

VOC_COCO_CLASS_NAMES["TOWOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

SUPERCLASS_NAMES = {
    'TOWOD': {v: k for k in mowodb_t4_superclasses for v in mowodb_t4_superclasses[k]},
    'OWDETR': {v: k for k in sowodb_t4_superclasses for v in sowodb_t4_superclasses[k]}
}

SUPERCLASS_INDICES = {
    'TOWOD': {k:i for i,k in enumerate(mowodb_t4_superclasses.keys())},
    'OWDETR': {k:i for i,k in enumerate(sowodb_t4_superclasses.keys())}
}

SUP_CLS_IND_MATCH_DICTS = {
    'TOWOD': {k: [VOC_COCO_CLASS_NAMES["TOWOD"].index(c) for c in mowodb_t4_superclasses[k]] for k in mowodb_t4_superclasses},
    'OWDETR': {k: [VOC_COCO_CLASS_NAMES["OWDETR"].index(c) for c in sowodb_t4_superclasses[k]] for k in sowodb_t4_superclasses}
}


class OWDetection(VisionDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 image_set='train',
                 transforms=None,
                 filter_pct=-1,
                 dataset='OWDETR', 
                 pseudo_path='', 
                ):
                 
        super(OWDetection, self).__init__(transforms)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.str_imgids = []
        self.imgid2annotations = {}
        self.image_set = []
        self.transforms=transforms
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset]
        self.MAX_NUM_OBJECTS = args.max_num_objects
        self.args = args
        self.dataset=dataset

        self.MAX_UNK_OBJECTS = args.num_unk_objects
        self.unk_idx = len(self.CLASS_NAMES) - 1

        self.superclass_names = SUPERCLASS_NAMES[dataset]
        self.superclass_indices = SUPERCLASS_INDICES[dataset]

        self.known_classes = list(range(args.PREV_INTRODUCED_CLS, args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS))
        print('known_classes', self.known_classes)

        self.root=str(root)
        annotation_dir = os.path.join(self.root, 'Annotations')
        image_dir = os.path.join(self.root, 'JPEGImages')

        file_names = self.extract_fns(image_set, self.root)
        if image_set == 'voc2007_trainval':
            print('PASCAL-VOC2007 dataset used; clearing images with missing object classes')
            prev_intro_cls = self.args.PREV_INTRODUCED_CLS
            curr_intro_cls = self.args.CUR_INTRODUCED_CLS
            valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
            current_file_names=[]
            for file in file_names:
                annot = os.path.join(annotation_dir, file + ".xml")
                tree = ET.parse(annot)
                target = self.parse_voc_xml(tree.getroot())
                instances = []
                for obj in target['annotation']['object']:
                    cls = obj["name"]
                    if cls in VOC_CLASS_NAMES_COCOFIED:
                        cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
                    
                    if self.CLASS_NAMES.index(cls) in valid_classes:
                        instance = dict(
                            category_id=self.CLASS_NAMES.index(cls),
                        )
                        instances.append(instance)
                if len(instances)>0:
                    current_file_names.append(file)

            self.image_set.extend(current_file_names)
            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in current_file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in current_file_names])
            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in current_file_names)
        else: 
            self.image_set.extend(file_names)
            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            self.str_imgids = file_names

        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        if pseudo_path != '':
            self.pseudo_nms_iou = args.pseudo_nms_iou
            self.pseudo_threshold = args.pseudo_threshold

            self.pseudo_coco = COCO(pseudo_path)

            if args.alternative_pseudo_path != '':
                self.alternative_pseudo_coco = COCO(args.alternative_pseudo_path)

        if args.pseudo_backup != '':
            self.pseudo_backup = COCO(args.pseudo_backup)

        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2021'):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 12 or len(x) == 6:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):

        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
    
        image_id = target['annotation']['filename']
        instances = []

        for obj_idx, obj in enumerate(target['annotation']['object']):

            cls = obj["name"]

            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            instance = dict(
                category_id=self.CLASS_NAMES.index(cls),
                superclass_id=self.superclass_indices[self.superclass_names[cls]] if self.CLASS_NAMES.index(cls) < self.args.PREV_INTRODUCED_CLS + self.args.CUR_INTRODUCED_CLS else self.superclass_indices['unknown'],
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
        return target, instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, self.dataset)
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes #81
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = self.unk_idx
                annotation["superclass_id"] = self.superclass_indices['unknown']
        return entry


    def merge_pseudo_instances_nms(self, target, str_imgid, orig_shape=None):

        pseudo_ann = self.pseudo_coco.load_anns(
            self.pseudo_coco.getAnnIds(imgIds=[str_imgid])
        )

        alternative_pseudo_ann = []
        if hasattr(self, 'alternative_pseudo_coco'):
            alternative_pseudo_ann = self.alternative_pseudo_coco.load_anns(
                self.alternative_pseudo_coco.getAnnIds(imgIds=[str_imgid])
            )
            pseudo_ann.extend(alternative_pseudo_ann)


        backup_pseudo_ann = []
        if hasattr(self, 'pseudo_backup'):
            backup_pseudo_ann = self.pseudo_backup.load_anns(
                self.pseudo_backup.getAnnIds(imgIds=[str_imgid])
            )
            pseudo_ann.extend(backup_pseudo_ann)


        if pseudo_ann == []:
            raise ValueError('No pseudo annotations found for image id {}'.format(str_imgid))

        boxes, scores, idx = [], [], []

        for t in target:
            boxes.append(t['bbox'])
            scores.append(1.0)
            idx.append(self.unk_idx)

        for ann in pseudo_ann:
            if ann['score'] < self.pseudo_threshold:
                continue
            x,y,w,h = ann['bbox']
            boxes.append([x,y,x+w,y+h])
            scores.append(ann['score'])
            idx.append(self.unk_idx)

        boxes = torch.as_tensor(boxes)
        scores = torch.as_tensor(scores)
        idx = torch.as_tensor(idx)

        max_num_objects = min((len(target) + self.MAX_UNK_OBJECTS), self.MAX_NUM_OBJECTS)

        keep = batched_nms(
            boxes = boxes, 
            scores = scores, 
            idxs = idx, 
            iou_threshold = self.pseudo_nms_iou)[:max_num_objects]

        merged = []

        for k in keep:
            cat_id = idx[k] if k.item() >= len(target) else target[k.item()]['category_id']
            sup_id = self.superclass_indices[self.superclass_names[self.CLASS_NAMES[cat_id]]] if cat_id != self.unk_idx else self.superclass_indices['unknown']

            merged.append(dict(
                category_id = cat_id,
                superclass_id = sup_id,
                bbox = list(boxes[k]),
                area = (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]),
                image_id = str_imgid
            ))

        return merged

    def __getitem__(self, index):
        """
        Args:
            index (int): Indexin

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])
        w, h = map(target['annotation']['size'].get, ['width', 'height'])

        if 'train' in image_set:
            instances = self.remove_prev_class_and_unk_instances(instances)
            if hasattr(self, 'pseudo_coco'):
                instances = self.merge_pseudo_instances_nms(instances, str_imgid=self.str_imgids[index]) #, orig_shape=(int(w), int(h)))

        elif 'test' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances = self.remove_unknown_instances(instances)
            if hasattr(self, 'pseudo_coco'):
                instances = self.merge_pseudo_instances_nms(instances, str_imgid=self.str_imgids[index]) #, orig_shape=(int(w), int(h)))

        target = dict(
            image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            superclasses=torch.tensor([i['superclass_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8),
        )

        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)