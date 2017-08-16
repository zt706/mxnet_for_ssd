from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import json
from evaluate.eval_voc import voc_eval
import cv2
import StringIO
import PIL
from PIL import Image


class TupuFace(Imdb):
    """
    Implementation of Imdb for tupu face datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    devkit_path : str
        devkit path of VOC dataset
    json_list_path: str
        list json file path
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, json_list_path, shuffle=False, is_train=False,
            names='tupu_face.names'):
        super(TupuFace, self).__init__('tupu_face' + image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = json_list_path
        self.extension = '.jpg'
        self.is_train = is_train

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.label_raw_info = [] 
        self.im_size = []
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()


    def _check_broken_img(self, im_path):
        im = None
        im_shape = None
        is_broken = False
        try:
            im = cv2.imread(im_path)
            if im is None:
                print('im is none: %s' % (im_path))
                is_broken = True

            with open(im_path, 'rb') as img_bin:
                buff = StringIO.StringIO()
                buff.write(img_bin.read())
                buff.seek(0)
                temp_img = np.array(PIL.Image.open(buff), dtype=np.uint8)
                img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        except Exception, ex:
            if 'error return without exception set' in ex:
                print('img broken %s, ex: %s ' % (im_path, ex) )
            else:
                print('unknow exception %s, ex: %s ' % (im_path, ex))
            is_broken = True
        finally:
            if not is_broken:
                im_shape = im.shape
            return is_broken, im_shape

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = self.data_path
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        image_set_index = []
        with open(image_set_index_file) as f:
            lines = f.readlines()[1:]
            idx = 0
            for l in lines:
                im_name = l.split('\t')[0]
                is_broken, im_shape = self._check_broken_img(im_name)
                if is_broken:
                    continue
                image_set_index.append((idx, im_name))
                self.label_raw_info.append((idx, l.split('\t')[1].strip()))
                self.im_size.append((idx, im_shape))
                idx += 1
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        image_file = self.image_set_index[index][1]
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _size_from_index(self, index):
        """
        given image index, find out im size

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        size of img
        """
        size = self.im_size[index][1] 
        assert size is not '', 'size not exist: {}'.format(self.image_set_index[index])
        return size 

    def _label_info_from_index(self, index):
        """
        given image index, find out annotation raw info 

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        raw info of annotation 
        """
        raw_label_info = self.label_raw_info[index][1] 
        assert raw_label_info is not '', 'label raw info not exist: {}'.format(self.image_set_index[index])
        return raw_label_info

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from raw label annotations
        for idx,_ in self.image_set_index:
            raw_info = self._label_info_from_index(idx)
            raw_info = json.loads(raw_info)
            size = self._size_from_index(idx)
            width = float(size[1])
            height = float(size[0])
            label = []

            for obj in raw_info:
                difficult = int(obj['difficult'])
                # if not self.config['use_difficult'] and difficult == 1:
                #     continue
                cls_name = obj['name']
                if cls_name not in self.classes:
                    continue
                cls_id = self.classes.index(cls_name)
                xmin = float(obj['xmin']) / width
                ymin = float(obj['ymin']) / height
                xmax = float(obj['xmax']) / width
                ymax = float(obj['ymax']) / height
                label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
            temp.append(np.array(label))
        return temp

