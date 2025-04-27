import torch
import os
import pickle

@torch.no_grad()
def main():
    data_dir = "data" + os.sep + "Imagenet64"

    limit = -1
    if limit == -1:
        limit = len(os.listdir(data_dir))

    imgs = torch.zeros(limit, 3, 64, 64, dtype=torch.uint8)
    labels = torch.zeros(limit, dtype=torch.int32)
    cur_iter = 0

    for file in os.listdir(data_dir):
        filename = data_dir + os.sep + file

        try:
            tens = pickle.load(open(filename, "rb"))
        except pickle.UnpicklingError:
            print(f"Pickle error with file {filename}")
            continue

        try:
            img = tens["img"]
            label = tens["label"]
        except KeyError:
            print(f"Key error with file {filename}")
            continue

        try:
            img = torch.tensor(img, dtype=torch.uint8).reshape(3, 64, 64)
            label = torch.tensor(label, dtype=torch.int32)
        except RuntimeError:
            print(f"Shape error with file {filename}")
            continue

        imgs[cur_iter] = img
        labels[cur_iter] = label

        cur_iter += 1

        if cur_iter == limit:
            break
    

    imgs = imgs[:cur_iter]
    labels = labels[:cur_iter]

    torch.save(imgs, "data/Imagenet64_imgs.pt")
    torch.save(labels, "data/Imagenet64_labels.pt")


if __name__ == "__main__":
    main()