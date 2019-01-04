import time
import torch
import os


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))


def count_files(folder, extension):
    count = 0
    for r, d, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                count += 1

    return count


def collect_filenames(folder, extension):
    """Take the names from all files within the given folder, which are tif
    Args:
        folder (string): Path to folder containing tiff images.
        extension (string): Only select files matching this extension e.g. '.tif'

    Return:
        files (list): Sorted list of file names with absolute path.
    """
    files = [os.path.join(folder, entry.name) for entry in os.scandir(folder)
             if entry.is_file() and entry.name.endswith(extension)]
    return sorted(files)


def contains_nan(x):
    """Check if torch.tensor has any NaN value.
    Args:
        x (torch.Tensor): tensor of any type

    Return: True if minimum 0ne NaN value was found
    """
    return torch.isnan(x).nonzero().nelement() != 0


class SavePointManager(object):
    """ Manage the number of stored checkpoints.
    """
    def __init__(self, path, max_chpt):
        self.path = path
        self.max_num_chpt = max_chpt
        self.files = dict()

    def save(self, model_state, filename, loss):
        """ Store model state, if loss is smaller than previous results or we did not reach the maximum allowed
         number of model checkpoints.

        :return: True if state was stored
        """
        saved_value = False
        if len(self.files) < self.max_num_chpt:
            torch.save(model_state, os.path.join(self.path, filename))
            self.files[loss] = filename
            saved_value = True
        elif loss < max(self.files):
            torch.save(model_state, os.path.join(self.path, filename))
            os.remove(self.files.pop(max(self.files)))
            self.files[loss] = filename
            saved_value = True

        return saved_value
