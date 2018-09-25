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
    files = [os.path.join(folder, entry.name) for entry in os.scandir(folder) if
              entry.is_file() and entry.name.endswith(extension)]
    return sorted(files)
