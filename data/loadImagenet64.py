import zipfile
import os
import pickle


if not os.path.exists("Imagenet64"):
    os.mkdir("Imagenet64")

idx_ctr = 0

unique_cls = set()

for archive in ["Imagenet64_train_part1.zip", "Imagenet64_train_part2.zip"]:
    with zipfile.ZipFile(archive, 'r') as archive_ld:
        for filename in archive_ld.filelist:
            file = pickle.load(archive_ld.open(filename.filename, "r"))
            for (img, label) in zip(file["data"], file["labels"]):
                img_label = dict(
                    img=img,
                    label=label
                )
                unique_cls.add(label)
                with open(f"Imagenet64/{idx_ctr}.pkl", "wb") as f:
                    pickle.dump(img_label, f)
                idx_ctr += 1
            
            del file

with open(f"Imagenet64/metadata.pkl", "wb") as f:
    pickle.dump(dict(
        num_data=idx_ctr,
        cls_min=min(unique_cls),
        cls_max=max(unique_cls)
    ), f)