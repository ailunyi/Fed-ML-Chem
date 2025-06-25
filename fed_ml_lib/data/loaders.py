from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, Dataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import pickle

from fed_ml_lib.core.utils import *
from .setup import *
from .partitioning import *

"""
This file contains the loaders for the different tasks. (copied from data_setup.py)
"""

# Normalization values for different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'cifar10': dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'imagenet': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'PILL': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'Wafer': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'BREAST': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'HISTO': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'MRI': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'PCOS': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'MMF': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'DNA+MRI' : dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'kidney_ct': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Medical imaging normalization
    'HIV': None,  
    'DNA': None
}

class MultimodalDataset(Dataset):
    def __init__(self, dna_dataset, mri_dataset):
        self.dna_dataset = dna_dataset
        self.mri_dataset = mri_dataset
        self.length = min(len(self.dna_dataset), len(self.mri_dataset))
        self.dna_indices = list(range(len(self.dna_dataset)))
        self.mri_indices = list(range(len(self.mri_dataset)))
        if len(self.dna_dataset) > self.length:
            self.dna_indices = random.sample(self.dna_indices, self.length)
        if len(self.mri_dataset) > self.length:
            self.mri_indices = random.sample(self.mri_indices, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dna_data, dna_label = self.dna_dataset[self.dna_indices[idx]]
        mri_data, mri_label = self.mri_dataset[self.mri_indices[idx]]
        
        return (mri_data, dna_data), (mri_label, dna_label)
    

def read_and_prepare_data(file_path, seed, size=6):
    """
    Reads DNA sequence data from a text file and prepares it for modeling.
    """
    # Read data from file
    data = pd.read_table(file_path)

    # Function to extract k-mers from a sequence
    def getKmers(sequence):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    # Function to preprocess data
    def preprocess_data(data):
        data['words'] = data['sequence'].apply(lambda x: getKmers(x))
        data = data.drop('sequence', axis=1)
        return data

    # Preprocess data
    data = preprocess_data(data)

    def kmer_lists_to_texts(kmer_lists):
        return [' '.join(map(str, l)) for l in kmer_lists]

    data['texts'] = kmer_lists_to_texts(data['words'])

    # Prepare data for modeling
    texts = data['texts'].tolist()
    y_data = data.iloc[:, 0].values
    model = TfidfVectorizer(ngram_range=(5,5))
    embeddings = model.fit_transform(texts).toarray()
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_data, test_size=0.2, random_state=seed)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    return trainset, testset

def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val=""):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param resize: the size of the image after resizing (if None, no resizing)
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """
    DataLoader = PyGDataLoader if dataset == "hiv" else TorchDataLoader
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])] if dataset not in ["MMF", "DNA", "hiv"] else None
    print(dataset)

    if dataset == "cifar":
        # Download and transform CIFAR-10 (train and test)
        transformer = transforms.Compose(
            list_transforms
        )
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)
    
    elif dataset == "hiv":
        trainset, testset = preprocess_graph()    

    elif dataset == "DNA":
        trainset, testset = read_and_prepare_data(data_path + dataset + '/human.txt', seed)        

    elif dataset == "MMF":        
        trainset, valset, testset = preprocess_and_split_data(data_path + dataset + '/Audio_Vision_RAVDESS.pkl')                    
    
    elif dataset == "DNA+MRI":        
        dataset_dna, dataset_mri = dataset.split("+")         
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms            

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset_mri)
        supp_ds_store(data_path + dataset_mri + "/Training")
        supp_ds_store(data_path + dataset_mri + "/Testing")
        trainset_mri = datasets.ImageFolder(data_path + dataset_mri + "/Training", transform=transformer)
        testset_mri = datasets.ImageFolder(data_path + dataset_mri + "/Testing", transform=transformer)
        trainset_dna, testset_dna = read_and_prepare_data(data_path + dataset_dna + '/human.txt', seed)
        trainset = MultimodalDataset(trainset_dna, trainset_mri)
        testset = MultimodalDataset(testset_dna , testset_mri)

    else:
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset + "/Training")
        supp_ds_store(data_path + dataset + "/Testing")
        trainset = datasets.ImageFolder(data_path + dataset + "/Training", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/Testing", transform=transformer)

    if dataset == "DNA":
        print("The training set is created for the classes: ('0', '1', '2', '3', '4', '5', '6')")
    elif dataset == "MMF":
        print("The training set is created for the classes: ('happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust', 'calm', 'neutral')")        
    elif dataset == "DNA+MRI":
        print("The training set is created for the classes: ")
        print("('glioma', 'meningioma', 'notumor', 'pituitary')")
        print("('0', '1', '2', '3', '4', '5', '6')")
    elif dataset == "kidney_ct":
        print("The training set is created for the classes: ('Cyst', 'Normal', 'Stone', 'Tumor')")
        print("Binary classification: Normal (0) vs Malignant (1) - Cyst/Stone/Tumor")
    elif dataset == "hiv":
        print("The training set is created for the classes: ('confirmed inactive (CI)', 'confirmed active (CA)/confirmed moderately active (CM)')")
    else:
        print(f"The training set is created for the classes : {trainset.classes}")        

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets_train = split_data_client(trainset, num_clients, seed)
    if dataset == "MMF":
        datasets_val = split_data_client(valset, num_clients, seed)
    elif data_path_val and dataset not in ["DNA", "hiv"]:
        valset = datasets.ImageFolder(data_path_val, transform=transformer)
        datasets_val = split_data_client(valset, num_clients, seed)    
        
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if dataset == "MMF" or data_path_val:
            # Use provided validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=dataset != "MMF"))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size))
        else:            
            len_val = int(len(datasets_train[i]) * splitter / 100)  # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed)) 
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader

def create_data_loaders(dataset_name: str, data_path: str = "./data/", 
                       batch_size: int = 32, resize: int = None, 
                       seed: int = 42, num_workers: int = 0, 
                       test_split: float = 0.2):
    """
    Create train and test data loaders for a given dataset.
    This is a simplified wrapper around load_datasets for single-machine use.
    """
    # For single machine, we just need 1 client
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=1, 
        batch_size=batch_size, 
        resize=resize, 
        seed=seed, 
        num_workers=num_workers,
        splitter=int(test_split * 100),
        dataset=dataset_name,
        data_path=data_path
    )
    
    return trainloaders[0], valloaders[0], testloader

def create_federated_data_loaders(dataset_name: str, num_clients: int,
                                 data_path: str = "./data/", 
                                 batch_size: int = 32, resize: int = None,
                                 seed: int = 42, num_workers: int = 0,
                                 partition_type: str = "iid"):
    """
    Create federated data loaders for multiple clients.
    """
    return load_datasets(
        num_clients=num_clients,
        batch_size=batch_size,
        resize=resize,
        seed=seed,
        num_workers=num_workers,
        dataset=dataset_name,
        data_path=data_path
    )

def get_dataset_info(dataset_name: str, data_path: str = "./data/"):
    """
    Get information about a dataset.
    """
    info = {
        'name': dataset_name,
        'path': data_path,
        'normalize_dict': NORMALIZE_DICT.get(dataset_name, None)
    }
    
    if dataset_name == "DNA":
        info['num_classes'] = 7
        info['input_shape'] = (None,)  # Variable length sequences
        info['description'] = "DNA sequence classification dataset"
    elif dataset_name == "MMF":
        info['num_classes'] = 8
        info['input_shape'] = (None,)  # Variable features
        info['description'] = "Multimodal Audio-Vision emotion recognition dataset"
    elif dataset_name == "hiv":
        info['num_classes'] = 2
        info['input_shape'] = (None,)  # Graph data
        info['description'] = "HIV molecule classification dataset"
    elif dataset_name in ["PILL", "MRI", "BREAST", "HISTO", "PCOS", "Wafer"]:
        info['num_classes'] = None  # Will be determined from dataset
        info['input_shape'] = (3, 224, 224)  # Default image size
        info['description'] = f"{dataset_name} image classification dataset"
    else:
        info['num_classes'] = None
        info['input_shape'] = None
        info['description'] = f"Unknown dataset: {dataset_name}"
    
    return info

def get_transforms(dataset_name: str, resize: int = None):
    """
    Get transforms for a given dataset.
    """
    if dataset_name in ["MMF", "DNA", "hiv"]:
        return None  # These datasets don't use image transforms
    
    normalize_params = NORMALIZE_DICT.get(dataset_name, NORMALIZE_DICT['imagenet'])
    
    transform_list = []
    if resize is not None:
        transform_list.append(transforms.Resize((resize, resize)))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(**normalize_params)
    ])
    
    return transforms.Compose(transform_list)
