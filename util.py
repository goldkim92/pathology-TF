import re
import openpyxl
import numpy as np
import scipy.misc as scm


def get_image(img_path, input_size, phase='train'):
    img = scm.imread(img_path)
    img = img[:,:,:3]
    if phase in ['train','valid']:
        i_start, j_start = (np.random.random(2) * 112).astype(int)
        img = img[i_start:i_start+input_size, j_start:j_start+input_size,:]
    else: # phase == 'test'
        i_start, j_start = 30,30;
        img = img[i_start:i_start+input_size, j_start:j_start+input_size,:]
                
    img = img/127.5 - 1.
    
    if phase == 'train':
        rand = np.random.random()
        if rand >= 0.666:
            img = np.flip(img,1)
        elif rand >= 0.333:
            img = np.flip(img,0)
            
    return img

def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)

def get_labels(label_path):
    prognosis_group = {
        'good':[1,0,0],
        'intermediate':[0,1,0],
        'bad':[0,0,1]
    }
    labels_dic = {}

    wb = openpyxl.load_workbook(label_path)
    sheet = wb.active

    multiple_cells = sheet['A2':'F374']
    for row in multiple_cells:
        name = row[0].value
        prognosis = row[-1].value
        if prognosis not in prognosis_group.keys():
            continue
        else:    
            prognosis = prognosis_group[prognosis]

        labels_dic[name] = prognosis
    
    return labels_dic

# Sort a string with a number inside
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

