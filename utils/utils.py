import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop


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


def encode_label_idda_dice(label):
    class_map = [11, 1, 4, 11, 5, 3, 6, 6, 7, 10, 2, 11, 8, 11, 11, 11, 0, 11, 11, 11, 9, 11, 11, 11, 1, 11, 11]

    label = np.array(label)

    # Build the first layer of the encoded image (void class)
    ohe_image = np.zeros(label.shape[:2] + (12,))

    idda_classes = 27

    for i in range(idda_classes):
        depth_class_color = (i, 0, 0)
        class_mask = np.equal(label, depth_class_color).all(axis=2)
        ohe_image[:, :, class_map[i]] += np.ones(class_mask.shape) * class_mask

    return np.transpose(ohe_image, (2, 0, 1))


def encode_label_idda_crossentropy(label):
    class_map = [11, 1, 4, 11, 5, 3, 6, 6, 7, 10, 2, 11, 8, 11, 11, 11, 0, 11, 11, 11, 9, 11, 11, 11, 1, 11, 11]

    label = np.array(label)
    # Build the first layer of the encoded image (void class)
    return np.array([class_map[v] for v in np.max(label, axis=-1).flatten()]).reshape(label.shape[:2])


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
    x = torch.argmax(image, dim=0)
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
    '''
    N.B. 
    pred comes from pred=torch.argmax(pred), that moves the value range to 0->11 (because we have 12 layers, 1 per class)
    
    The way this piece of code is working is really intricate but smart. 
    1) The code starts by creating a mask for predicted values that are not void (active classes)
    2) After that, num_classes*pred[mask] shifts the classes apart of num_classes, which is wide enough to accept 
    num_classes different values for each class (num_classes squared)
    3) Finally, label[mask] gets summed to the previous value, this piece of information tells us which was the true
    class associated with that prediction.
    4) The vector is then run through bincount, which should yield one count for each value. Due to the sorted nature
    of how the problem is built, specifying the minlength parameter is enough to account for those combinations that 
    never appear
    5) Reshaping the array to (num_classes, num_classes) essentially gives us a confusion matrix.
    '''

    # Mask for pixels which have a class non void
    mask = (pred >= 0) & (pred < num_classes)

    return torch.bincount(num_classes * pred[mask] + label[mask], minlength=num_classes**2).view(num_classes, num_classes).detach().cpu()


def intersection_over_union(matrix, epsilon=1e-5):
    """
    Computes Per-Class Intersection-Over-Union metric. (Jaccard Index)
    IoU computes the ratio between the:
        - intersection (of prediction and truth, for class C): number of pixels assigned to class C in both images.
    and the
        - union (of prediction and truth, fod class C): number of pixel assigned to class C in either images - inters.

    Args:
        matrix (np.array): Confusion matrix of the prediction with respect to the label.
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

    # correct_per_class = np.diag(matrix)
    # total_per_class = matrix.sum(axis=1)
    iou = np.diag(matrix) / (matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix) + epsilon)

    return iou[:-1], torch.mean(iou[:-1])


def seed_worker(worker_id):
    """
    Sets the data loader workers seeds for reproducibility purposes

    Args:
        worker_id:

    Returns:
        Nothing
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def convert_class_to_color(img):
    class_to_color = [(0, 128, 192), (128, 0, 0), (64, 0, 128), (192, 192, 128), (64, 64, 128), (64, 64, 0),
                      (128, 64, 128), (0, 0, 192), (192, 128, 128), (128, 128, 128), (128, 128, 0), (0, 0, 0)]

    new_img = np.zeros(img.shape + (3,), dtype=np.uint8)

    img = img.detach().cpu().numpy()
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            try:
                color = class_to_color[img[r, c]]
            except Exception as e:
                color = 0
            new_img[r, c] = color

    return new_img


def plot_prediction(model, dataloader_val, epoch, dataset='CamVid'):
    if dataset == 'CamVid':
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test/Seq05VD_f01110.png'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/CamVid/test_labels/Seq05VD_f01110_L.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_dice(Image.open(label_path), get_label_info('data/CamVid/class_dict.csv')))

    else:
        image_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test/@277531.725@110474.014@Town10@ClearNoon@audi@1608329311@0.998879709441141@1.000840205377106@0.7391348481178284@242616@.jpg'
        label_path = '/home/nicola/Documents/uni/MLDL/project/BiSeNet/data/IDDA/test_labels/@277531.725@110474.014@Town10@ClearNoon@audi@1608329311@0.998879709441141@1.000840205377106@0.7391348481178284@242616@.png'
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        label = torch.tensor(encode_label_idda_dice(np.array(Image.open(label_path).convert('RGB'))))

    image = normalize(Image.open(image_path))
    model.eval()
    with torch.no_grad():

        image, label = image.cuda(), torch.tensor(label.cuda())
        fig, ax = plt.subplots(1, 2)
        ax = ax.ravel()
        predict = model(image.unsqueeze(0))
        ax[0].imshow(convert_class_to_color(reverse_one_hot(predict[0])))
        ax[0].set_title("predicted")

        ax[1].imshow(convert_class_to_color(reverse_one_hot(label)))
        ax[1].set_title("label")
        plt.savefig(f'images/pred_{epoch}')
        plt.show()


def batch_to_plottable_image(batch):
    return np.moveaxis(np.array(batch[0].detach().cpu()), 0, -1)


def label_to_plottable_image(batch):
    return convert_class_to_color(reverse_one_hot(batch[0].detach().cpu()))


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
# return lr


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
