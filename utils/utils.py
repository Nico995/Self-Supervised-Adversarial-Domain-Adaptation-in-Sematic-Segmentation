import random
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
        void_index (int): void class index

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


def global_accuracy(pred, label):
    """
    Compute Pixel Accuracy metric, i.e. the percentage of corresponding pixels between the prediction and the label.

    Args:
        pred (tensor): Predicted segmentation from Model.
        label (tensor): Label image from Dataloader.

    Returns:
        Per pixel accuracy
    """

    pred = pred.flatten()
    label = label.flatten()

    total = len(label)
    count = (pred == label).sum()
    return float(count) / float(total)


def get_confusion_matrix(pred, label, num_classes):
    """
    Computes the confusion matrix

    Args:
        pred: Predicted segmentation from Model.
        label: Label image from Dataloader.
        num_classes: Total classes for the current classification task.

    Returns:
        Confusion matrix
    """
    # Mask for pixels which have a class non void
    mask = (label >= 0) & (label < num_classes)

    # Computes the confusion matrix for the classes
    return np.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


def intersection_over_union(matrix, metadata, epsilon=1e-5):
    """
    Computes Per-Class Intersection-Over-Union metric. (Jaccard Index)
    IoU computes the ratio between the:
        - intersection (of prediction and truth, for class C): number of pixels assigned to class C in both images.
    and the
        - union (of prediction and truth, fod class C): number of pixel assigned to class C in either images - inters.

    Args:
        matrix (np.array): Confusion matrix of the prediction with respect to the label.
        metadata (str): Path for metedata csv
        epsilon (float): small value used to avoid divisions by 0, in case the class is not present in the prediction.
    Returns:
        Per-Class Intersection-Over-Union
        Mean Per-Class Intersection-Over-Union
    """

    '''
    - Numerator: torch.diag(confusion_matrix):
        On the diagonal of the conf matrix we have the correctly classified pixels byt both images (->intersection)
    - Denominator: matrix.sum(1) - np.diag(confusion_matrix) + epsilon
        Sum of the pixel classified as C by either images (-> union)(sum 1) minus the intersection
        
    In both cases, epsilon is used to avoid numerical overflow.
    '''

    correct_per_class = np.diag(matrix)
    total_per_class = matrix.sum(axis=1)
    ious = (correct_per_class + epsilon) / (2 * total_per_class - correct_per_class + epsilon)

    df = pd.read_csv(metadata)
    names = df.loc[df["class_11"] == 1, "name"]

    return zip(names, ious), np.mean(ious)


def seed_worker(worker_id):
    """
    Sets the data loader workers seeds for reproducibility purposes

    Args:
        worker_id:

    Returns:
        Nothing
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def convert_class_to_color(img):
    class_to_color = [(0, 0, 0), (0, 128, 192), (128, 0, 0), (64, 0, 128), (192, 192, 128), (64, 64, 128), (64, 64, 0),
                      (128, 64, 128), (0, 0, 192), (192, 128, 128), (128, 128, 128), (128, 128, 0)]

    new_img = np.zeros(img.shape + (3,), dtype=np.uint8)

    img = img.detach().cpu().numpy()
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            color = class_to_color[img[r, c]]
            new_img[r, c] = color

    return new_img
