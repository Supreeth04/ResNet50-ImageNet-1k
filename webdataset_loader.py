import webdataset as wds

def create_webdataset_loader(urls, batch_size):
    dataset = (
        wds.WebDataset(urls)
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "cls")
        .map_tuple(transform, lambda x: x)
        .batched(batch_size)
    )
    return dataset 