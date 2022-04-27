from dataset.train_dataset import GeneralRendererDataset, FinetuningRendererDataset

name2dataset={
    'gen': GeneralRendererDataset,
    'ft': FinetuningRendererDataset,
}