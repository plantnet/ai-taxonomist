"""
Create the ground truth for a trained ai-taxonomist dataset
Use --help!
"""
import argparse
import json
import os
import pygbif
import csv


def main():
    parser = argparse.ArgumentParser(description="build c4c-identify ground truth")
    parser.add_argument("-d", "--data", help="input/output directory", required=True)
    parser.add_argument("--images", default=None, help="list of images urls")
    parser.add_argument(
        "--name-type",
        type=str,
        choices=["canonicalName", "scientificName"],
        default="canonicalName",
        help="Use canonical or scientific names",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    args = parser.parse_args()

    if not args.images:
        args.images = os.path.join(args.data, "temporary", "images.csv")

    taxonomist_dir = os.path.join(args.data, "ai-taxonomist")
    gt_path = os.path.join(taxonomist_dir, "GT")
    os.makedirs(gt_path, exist_ok=True)

    id2class = {}
    species_id = []
    print("=> loading network mapping")
    with open(os.path.join(taxonomist_dir, "network", "mapping.json")) as f:
        id2class = json.load(f)

    print("=> retrieving names from gbif and building class mapping")
    mapping = [{} for i in range(len(id2class))]
    for k, v in id2class.items():
        # name_usage = pygbif.species.name_usage(key=k, data='name')
        name_usage = pygbif.species.name_usage(key=k)
        mapping[v] = {
            "name": name_usage.get(args.name_type, None),
            "species_id": k,
            "rank": name_usage.get("rank", None),
            "authorship": name_usage.get("authorship", None),
            "vernacularName": name_usage.get("vernacularName", None),
            "species": name_usage.get("species", None),
            "genus": name_usage.get("genus", None),
            "family": name_usage.get("family", None),
        }
    with open(os.path.join(gt_path, "classes.json"), "w") as f:
        json.dump(mapping, f)

    print("=> loading images urls and building images mapping")
    images = dict()
    with open(args.images) as f:
        urls = csv.DictReader(f)
        print(urls.fieldnames)
        for img in urls:
            basename = img.get("basename", None)
            species_id = img.get("label", None)
            if basename and species_id:
                class_id = id2class.get(species_id, -1)
                if class_id > 0:
                    images[basename] = img
                    images[basename]["class_id"] = class_id
    with open(os.path.join(gt_path, "images.json"), "w") as f:
        json.dump(images, f)


if __name__ == "__main__":
    main()
