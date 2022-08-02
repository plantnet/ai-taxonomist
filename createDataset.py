"""
Create a dataset for ai-taxonomist
Use --help!
"""
import shutil
import sys
import typing
import argparse
import json
import os
import gbif_dl
import pygbif.species as gbif_species
from img2dataset import download
import random


def binomials2species_ids(names: typing.TextIO, temp_data: str, verbose: int) -> list:
    """
    Search GBIF to convert a list of binomial names to the corresponding list of GBIF species_id
    :param names: an opened text file with on binomial per line
    :param temp_data: path of the output directory
    :param verbose: verbosity level
    :return: list of matching GBIF species_id
    """
    species_ids = []
    other = []
    cnt = 0
    for n in names:
        n = n.rstrip()
        data = gbif_species.name_backbone(name=n, rank="species", verbose=verbose > 1)
        match = data.get("matchType", "unknown")
        rank = data.get("rank", None)
        species_id = data.get("speciesKey", None)

        if match == "EXACT" and rank == "SPECIES" and species_id:
            species_ids.append(species_id)
        elif verbose:
            print(
                n,
                "does not match any species in gbif",
                {"name": n, "rank": rank, "match": match},
            )
            if verbose > 1:
                print(data)
            other.append({"name": n, "gbif": data})
        cnt += 1
        if verbose and cnt % 50 == 0:
            print(cnt, "species matched")

    if len(other):
        output = os.path.join(temp_data, "no_species.json")
        if verbose:
            print(
                "some species names do not match any species, see",
                output,
                "for details",
            )
        with open(output, "w") as f:
            json.dump(other, f)

    species_file = os.path.join(temp_data, "species_ids.json")
    with open(species_file, "w") as f:
        json.dump(species_ids, f)
        if verbose:
            print(len(species_ids), "gbif species ids stored in", species_file)

    return species_ids


def main():
    parser = argparse.ArgumentParser(description="ai-taxonomist dataset creation")
    required = parser.add_argument_group("required argument")
    required.add_argument(
        "-d",
        "--data",
        help="where to store dataset information and images",
        required=True,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_const",
        const=0,
        default=1,
        help="run silently",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_const",
        const=2,
        default=1,
        help="augment verbosity level",
    )
    crawl = parser.add_argument_group(title="crawl", description="control GBIF crawl")
    crawl_xor = crawl.add_mutually_exclusive_group(required=True)
    crawl_xor.add_argument(
        "--no-crawl",
        action="store_false",
        dest="crawl",
        help="do not crawl gbif, use the stored in DATA/temporary/images.csv instead",
    )
    crawl_xor.add_argument(
        "--doi",
        type=str,
        help="gbif Darwin Core Archive (dwca) doi, "
        "providing a prebuilt doi is the recommended way of using this tool",
    )

    crawl_xor.add_argument(
        "--names",
        type=argparse.FileType("r"),
        help="text file with one species binomial name per line",
    )
    crawl_xor.add_argument(
        "--species",
        type=argparse.FileType("r"),
        help="json file listing gbif species ids in an array",
    )
    crawl.add_argument(
        "--providers",
        type=argparse.FileType("r"),
        help="json file listing gbif providers ids in an array, applies to provided --name or --species",
    )
    crawl.add_argument(
        "-n",
        "--number",
        type=int,
        metavar="N",
        default=1000,
        help="Requested number of images per species, only applies with --name or --species",
    )

    dl = parser.add_argument_group(
        title="download", description="control images download"
    )
    dl.add_argument(
        "--no-download",
        action="store_false",
        dest="download",
        help="do not download images, use images stored in DATA/img/all",
    )
    dl.add_argument(
        "--workers", type=int, metavar="N", default=10, help="number of // downloads"
    )
    dl.add_argument(
        "--size", type=int, metavar="N", default=300, help="target image size"
    )

    split = parser.add_argument_group(
        title="split", description="control the train/val split"
    )
    split.add_argument(
        "--no-split",
        action="store_false",
        dest="split",
        help="do not split images in DATA/img/all in train and val sets",
    )
    split.add_argument(
        "--percent",
        type=float,
        default=0.01,
        metavar="P",
        help="keep P*number_of_images for the validation",
    )

    args = parser.parse_args()

    train_data = args.data
    temp_data = os.path.join(args.data, "temporary")
    for d in [train_data, temp_data]:
        os.makedirs(d, exist_ok=True)

    images_file = os.path.join(temp_data, "images.csv")
    if args.crawl:
        if args.verbose:
            print("Crawling gbif")
        if args.doi:
            # use a precomputed gbif query from its doi
            # save the doi for online usage
            doi = {"gbif_doi": args.doi}
            with open(os.path.join(temp_data, "doi.json"), "w") as f:
                json.dump(doi, f)

            data_generator = gbif_dl.dwca.generate_urls(
                args.doi,
                dwca_root_path=os.path.join(temp_data, "dwcas"),
                label="speciesKey",
                mediatype="StillImage",
                one_media_per_occurrence=False,
            )
        else:
            # retrieve media from a name or a species list and an optional providers list
            if args.names:
                species_id = binomials2species_ids(
                    args.names, temp_data, verbose=args.verbose
                )
            elif args.species:
                species_ids = json.load(args.species)

            if args.providers:
                providers = json.load(args.providers)
            else:
                providers = None
            queries = {"speciesKey": species_ids, "datasetKey": providers}
            data_generator = gbif_dl.api.generate_urls(
                queries=queries,
                label="speciesKey",
                nb_samples_per_stream=args.number,
                split_streams_by="speciesKey",
                mediatype="StillImage",
                one_media_per_occurrence=False,
                verbose=args.verbose > 1,
            )

        if args.verbose:
            print("Retrieving images metadata...")
        gbif_dl.export.to_csv(data_generator, images_file)
        if args.verbose:
            print("Images metadata saved in", images_file)
    else:
        if args.verbose:
            print("Crawl skipped, loading stored metadata")

    img_dir = os.path.join(train_data, "img")
    all_img_dir = os.path.join(img_dir, "all")
    if args.download:
        if args.verbose:
            print("Downloading images to", img_dir)
        download(
            url_list=images_file,
            output_folder=all_img_dir,
            input_format="csv",
            image_size=args.size,
            thread_count=args.workers,
            resize_mode="center_crop",
            extract_exif=False,
            save_additional_columns=["label", "basename"],
        )
        if args.verbose:
            print("Images downloaded")
    else:
        if args.verbose:
            print("Download skipped")

    if args.split:
        if args.verbose:
            print("Splitting images into train/val sets")
        per_label = {}
        img_cnt = 0
        for x in os.walk(all_img_dir):
            d = x[0]
            for file in x[2]:
                if file.endswith(".json") and not file.endswith("_stats.json"):
                    file_name = os.path.join(d, file)
                    with open(file_name, "r") as f:
                        info = json.load(f)
                        status = info.get("status", "failed")
                        label = info.get("label", None)
                        key = info.get("key", None)
                        basename = info.get("basename", None)
                        if status == "success" and label and key and basename:
                            per_label.setdefault(label, []).append(
                                {"basename": basename, "key": key, "dir": d}
                            )
                            img_cnt += 1
        if args.verbose:
            print("Found", img_cnt, "images illustrating", len(per_label), "species")
        train_cnt = 0
        val_cnt = 0
        for label, images in per_label.items():
            train_dir = os.path.join(img_dir, "train", str(label))
            val_dir = os.path.join(img_dir, "val", str(label))
            for d in [train_dir, val_dir]:
                os.makedirs(d, exist_ok=True)
            random.shuffle(images)
            n = len(images)
            nb_val = int(n * args.percent + 0.5)
            for i in images[:nb_val]:
                os.rename(
                    os.path.join(i["dir"], i["key"] + ".jpg"),
                    os.path.join(val_dir, i["basename"] + ".jpg"),
                )
                val_cnt += 1
            for i in images[nb_val:]:
                os.rename(
                    os.path.join(i["dir"], i["key"] + ".jpg"),
                    os.path.join(train_dir, i["basename"] + ".jpg"),
                )
                train_cnt += 1
        shutil.rmtree(all_img_dir, ignore_errors=True)
        if args.verbose:
            print(
                "Created a dataset with",
                train_cnt,
                "train and",
                val_cnt,
                "validation images illustrating",
                len(per_label),
                "species",
            )


if __name__ == "__main__":
    main()
