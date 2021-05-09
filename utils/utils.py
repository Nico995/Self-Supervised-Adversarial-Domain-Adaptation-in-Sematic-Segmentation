import numpy as np
import pandas as pd
import torch


def poly_lr_scheduler(optimizer, starting_lr, current_iter, max_iter=300, power=0.9):
    """
    Polynomial decay learning rate scheduler. Updates the optimizers
    parameters automatically, with the new learning rate.

    Args:
        optimizer: Optimizer used in training.
        starting_lr (float): Base learning rate to start the schedule with.
        current_iter (int): Current epoch in the training loop.
        max_iter (int): Final epoch in the training loop.
        power (float): Power to use in the polynomial computation.

    Returns:
        Current learning rate
    """
    lr = starting_lr * (1 - current_iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def get_label_info(csv_path):
    """
    Reads the labels metadata from the CSV into a python dictionary.
    Example row:
        Class_name, R,  G,  B, class11_flag
        Pedestrian, 64, 64, 0,  1

    Args:
        csv_path (str): Path of the CSV metadata.

    Returns:
        A python dictionary containing the parsed metadata.
    """

    ann = pd.read_csv(csv_path)
    label = {}
    for _, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_11 = row['class_11']
        label[label_name] = [int(r), int(g), int(b), class_11]
    return label


def encode_label_crossentropy(label, label_info, void_index=11):
    """
    Encodes the label images for Crossentropy Loss.
    Each label is an RGB image with a unique color for each class.
    This function changes the 3 channels color (R,G,B) to a single value representing the class index.

    The classes that don't have the value set to 1 in label_info end up in the void class (first channel)

    Args:
        label (Image): current label image to encode.
        label_info (dict): dictionary containing info on each class.

    Returns:
        ohe_image (np.array): encoded image
    """

    # Convert PIL.Image to np.array
    label = np.array(label)

    # Build the first layer of the encoded image (void class)
    ohe_image = np.zeros(label.shape[:2])

    class_count = 0
    # Iterate over all classes info (R, G, B, class11_flag)
    for index, info in enumerate(label_info):
        # Split color info from class11 flag
        color = label_info[info][:3]
        class11_flag = label_info[info][3]

        # get a mask of triplets, where each elements is True if the
        # current channel value corresponds to the current class info
        equality = np.equal(label, color)

        # Class matches only if all 3 channels match
        class_map = np.all(equality, axis=-1)

        # If class is in the 11_classes problem (or class11 problem is not considered at all)
        if class11_flag:
            # Set each pixel to the corresponding class index
            ohe_image[class_map] = class_count
            class_count += 1
        else:
            ohe_image[class_map] = void_index

    return ohe_image


def encode_label_dice(label, label_info):
    """
    Encodes the label images for Dice Loss.
    Each label is an RGB image with a unique color for each class.
    This function moves each class in a dedicated channel of the image, putting to 1 the pixels where that specific
    class is present (uses the RGB values to check for class presence).

    The classes that don't have the value set to 1 in label_info end up in the void class (first channel)

    Args:
        label (Image): current label image to encode.
        label_info (dict): dictionary containing info on each class.

    Returns:
        ohe_image (np.array): encoded image
    """

    # Convert PIL.Image to np.array
    label = np.array(label)

    # Build the first layer of the encoded image (void class)
    ohe_image = np.zeros(label.shape[:2] + (1,))

    # Iterate over all classes info (R, G, B, class11_flag)
    for index, info in enumerate(label_info):

        # Split color info from class11 flag
        color = label_info[info][:3]
        class11_flag = label_info[info][3]

        # get a mask of triplets, where each elements is True if the
        # current channel value corresponds to the current class info
        equality = np.equal(label, color)

        # Class matches only if all 3 channels match
        class_map = np.all(equality, axis=-1)
        # If class is in the 11_classes problem (or class11 problem is not considered at all)
        if class11_flag:
            # Create a new Channel for each class (one-hot like)
            ohe_image = np.dstack((ohe_image, class_map))
        else:
            # Fill the corresponding pixels in "void" channel (0)
            ohe_image[class_map, 0] = 1

    ohe_image = np.roll(np.moveaxis(ohe_image, -1, 0), shift=-1, axis=0)
    return ohe_image

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def cal_miou(miou_list, csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    miou_dict = {}
    cnt = 0
    for iter, row in ann.iterrows():
        label_name = row['name']
        class_11 = int(row['class_11'])
        if class_11 == 1:
            miou_dict[label_name] = miou_list[cnt]
            cnt += 1
    return miou_dict, np.mean(miou_list)
