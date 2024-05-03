import os
from shutil import copyfile


'''
Preprocessing the Market-1501 dataset, splitting it into train and test sets for feature extractor training 
'''


def dataset_split():
    dataset_path = './Market-1501-v15.09.15/Market-1501-v15.09.15'
    save_path = 'Market-1501'

    if not os.path.isdir(dataset_path):
        print('please change the path')

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # -----------------------------------------
    # query
    query_path = dataset_path + '/query'
    query_save_path = save_path + '/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    # -----------------------------------------
    # multi-query
    query_path = dataset_path + '/gt_bbox'
    # for dukemtmc-reid, we do not need multi-query
    if os.path.isdir(query_path):
        query_save_path = save_path + '/multi-query'
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = query_path + '/' + name
                dst_path = query_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

    # -----------------------------------------
    # gallery
    gallery_path = dataset_path + '/bounding_box_test'
    gallery_save_path = save_path + '/gallery'
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    # ---------------------------------------
    # train_all
    train_path = dataset_path + '/bounding_box_train'
    train_save_path = save_path + '/train_all'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    # ---------------------------------------
    # train_val
    train_path = dataset_path + '/bounding_box_train'
    train_save_path = save_path + '/train'
    val_save_path = save_path + '/test'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)


if __name__ == '__main__':
    dataset_split()
