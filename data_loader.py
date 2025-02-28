from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import torchvision.transforms as T

from typing import List, Tuple, Dict, Any


def get_domains(domains: str) -> Tuple[List[str], bool]:
    if "-" in domains:
        pool = False
        domains_list = domains.split("-")
    else:
        pool = True
        domains_list = domains.split("+")
    return domains_list, pool


class CMLDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.datasetname = hparams["dataset"]

        self.training_domains, self.pool_unlabeled = get_domains(self.hparams.training_domains)

        self.val_domains, self.pool_labeled = get_domains(self.hparams.val_domains)

        self.test_domains, self.pool_test = get_domains(self.hparams.test_domains)

        global CustomDataset
        if self.hparams.dataset in ["lumbarspine", "abdomen", "lung"]:
            from datasets.dataset import MIDataset as CustomDataset
            from datasets.dataset import NClASSES_dict
            self.num_classes = NClASSES_dict[self.hparams.dataset]
        else:
            raise NotImplementedError
    
    def load_dataset(self, domain: str, split: str, transformname: str = None):
        return CustomDataset(
            domain,
            path=self.hparams.path_in,
            split=split,
            dataset=self.datasetname,
            transformname=transformname,
            rgb_dim=self.hparams.rgb_dim,
            filter=self.hparams.filter,
            dataset_style=self.hparams.dataset_style
        )
    
    def setup(self, stage=None):
        print("Setting up data module...")

        if stage in (None, "fit"):
            train_dataset = []
            self.train_dataloader_len = 0
            self.train_dataloader_len_domain = []
            print("Setting up training domains...")
            for domain in self.training_domains:
                print(domain)
                train_dataset.append(
                    self.load_dataset(domain, "train", self.hparams.transforms)
                )
                self.train_dataloader_len += len(train_dataset[-1])
                self.train_dataloader_len_domain.append(len(train_dataset[-1]))
                print(f"Number of train points ({domain}): {len(train_dataset[-1]):,}")
                # local_counter = Counter(train_dataset[-1].dataset._y_array.numpy())
                # print(local_counter)

            val_dataset = []
            self.val_dataloader_len = 0
            print("Setting up val_domains domains...")
            for domain in self.val_domains:
                print(domain)
                val_dataset.append(
                    self.load_dataset(domain, "val", None)
                )
                self.val_dataloader_len += len(val_dataset[-1])
                print(f"Number of val points ({domain}): {len(val_dataset[-1]):,}")
                # local_counter = Counter(val_dataset[-1].dataset._y_array.numpy())
                # print(local_counter)

            if self.pool_unlabeled:
                self.train_dataset = [torch.utils.data.ConcatDataset(train_dataset)]
            else:
                self.train_dataset = train_dataset

            self.val_dataset = val_dataset

            self.num_train_dataloaders = len(self.train_dataset)

            print(f"Number of train points (total): {self.train_dataloader_len:,}")
            print(f"Number of val points (total): {self.val_dataloader_len:,}")

        if stage in (None, "test"):
            self.test_dataset = []
            for domain in self.test_domains:
                self.test_dataset.append(
                    self.load_dataset(domain, "test", None)
                )
                print(
                    f"Number of test points ({domain}): {len(self.test_dataset[-1]):,}"
                )
                # local_counter = Counter(self.test_dataset[-1].dataset._y_array.numpy())
                # print(local_counter)

            self.num_test_dataloaders = len(self.test_dataset)
            # self.domainsdict = self.test_dataset[-1].domainsdict

    def train_dataloader(self) -> List[DataLoader]:
        dataloaders = []
        for ds in self.train_dataset:
            dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    drop_last=True,
                    shuffle=True,
                )
            )
        print(len(dataloaders), "train dataloaders\n")
        if len(dataloaders) == 1:
            return dataloaders[0]
        else:
            return dataloaders

    def val_dataloader(self) -> List[DataLoader]:
        batch_size = self.hparams.batch_size
        shuffle = True

        dataloaders = []
        for ds in self.val_dataset:
            if self.hparams.depth_test:
                batch_size = ds.depth
                shuffle = False
            dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=self.hparams.num_workers,
                    drop_last=False,
                    shuffle=shuffle,
                )
            )
        print(len(dataloaders), "val dataloaders\n")
        if len(dataloaders) == 1:
            return dataloaders[0]
        else:
            return dataloaders

    def test_dataloader(self) -> List[DataLoader]:        
        batch_size = self.hparams.batch_size
        shuffle = True

        dataloaders = []        
        for ds in self.test_dataset:
            if self.hparams.depth_test:
                batch_size = ds.depth
                shuffle = False
            dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=self.hparams.num_workers,
                    drop_last=False,
                    shuffle=shuffle,
                )
            )
        print(len(dataloaders), "test dataloaders\n")
        return dataloaders


def setup_labeled_dataloader(hparams: Dict[str, Any]) -> DataLoader:
    print("Setting up labeled dataloader.")
    val_domains, _ = get_domains(hparams["val_domains"])

    datasets = []
    for domain in val_domains:
        print(domain)
        datasets.append(
            CustomDataset(
                domain,
                path=hparams["path_in"],
                split="train",
                dataset=hparams["dataset"],
                rgb_dim=hparams["rgb_dim"],
            )
        )

    datasets = torch.utils.data.ConcatDataset(datasets)

    print(f"Number of knn-dataset points: {len(datasets):,}")
    return DataLoader(
        datasets, drop_last=False, batch_size=hparams["batch_size"], shuffle=False
    )