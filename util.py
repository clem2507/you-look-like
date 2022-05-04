def is_face_centered(img, box):
    ''''
    Function that detects if the user face is centered enough on the frame to resize

    Parameters:

    -img: original image taken by the webcam
    -box: coordinates of the bounding box given by yolo

    Returns:

    -boolean: true if the user face is centered enough, else false
    '''

    tr_param = abs(abs(box['x_tl'] - box['x_br']) - abs(box['y_tl'] - box['y_br']))//2
    if abs(box['x_tl'] - box['x_br']) < abs(box['y_tl'] - box['y_br']):
        if (box['x_tl'] - tr_param > 0) and (box['x_br'] + tr_param < img.shape[1]):
            return True
        else:
            return False
    else:
        if (box['y_tl'] + tr_param < img.shape[0]) and (box['y_br'] - tr_param > 0):
            return True
        else:
            return False

def resize_face(img, box):
    ''''
    Method used to crop the user face in a pre-defined square size

    Parameters:

    -img: original image taken by the webcam
    -box: coordinates of the bounding box given by yolo

    Returns:

    -crop: the user face croped as expected
    '''

    import cv2

    new_box = {}
    tr_param = abs(abs(box['x_tl'] - box['x_br']) - abs(box['y_tl'] - box['y_br']))//2
    if abs(box['x_tl'] - box['x_br']) < abs(box['y_tl'] - box['y_br']):
        new_box = {'x_tl': box['x_tl'] - tr_param, 
                   'y_tl': box['y_tl'], 
                   'x_br': box['x_br'] + tr_param, 
                   'y_br': box['y_br']}
    else:
        new_box = {'x_tl': box['x_tl'], 
                   'y_tl': box['y_tl'] + tr_param, 
                   'x_br': box['x_br'], 
                   'y_br': box['y_br'] - tr_param}

    if abs(new_box['x_tl'] - new_box['x_br']) > abs(new_box['y_tl'] - new_box['y_br']):
        new_box['y_tl'] += abs(abs(new_box['x_tl'] - new_box['x_br']) - abs(new_box['y_tl'] - new_box['y_br']))
    if abs(new_box['x_tl'] - new_box['x_br']) < abs(new_box['y_tl'] - new_box['y_br']):
        new_box['x_br'] += abs(abs(new_box['x_tl'] - new_box['x_br']) - abs(new_box['y_tl'] - new_box['y_br']))

    crop_im = img[new_box['y_br']:new_box['y_tl'], new_box['x_tl']:new_box['x_br']]

    crop_im = cv2.resize(crop_im, (256, 256))
    
    return crop_im
