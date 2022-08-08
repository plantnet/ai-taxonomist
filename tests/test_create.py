import runpy
import os
import sys


def test_script_execution():
    script = os.path.join(__file__, "..", "..", "createDataset.py")
    data_dir = "nrtest"
    args = {
        "args": {"datap": data_dir, "doi": "10.15468/dl.j8xkgt", "percent": 0.1},
        "env": {"PYTHONPATH": os.path.join(__file__, "..", "..")},
    }
    try:
        sys.argv = [
            script,
            "--data",
            data_dir,
            "--doi",
            "10.15468/dl.j8xkgt",
            "--percent",
            "0.1",
        ]
        res = runpy.run_path(script, run_name="__main__")
        # res = runpy.run_path(script, init_globals=args)
    except SystemExit as e:
        print(e)
        assert 0
    except BaseException as e:
        print(e)
        assert 0

    # print(res)
    per_label = {}
    img_cnt = 0
    val_cnt = 0
    # allow rerun split by preloading per_label with existing train/val images if any
    train_dir = os.path.join(data_dir, "img", "train")
    val_dir = os.path.join(data_dir, "img", "val")
    for subdir in [train_dir, val_dir]:
        if os.path.isdir(subdir):
            for x in os.walk(subdir):
                d = x[0]
                label = os.path.basename(d)
                for file in x[2]:
                    basename = os.path.splitext(file)[0]
                    per_label.setdefault(label, dict())
                    if basename not in per_label[label].keys():
                        per_label[label][basename] = {"key": basename, "dir": d}
                        img_cnt += 1
                        if d.startswith(val_dir):
                            val_cnt += 1
                    else:
                        assert 0, "found duplicate image %s".format(basename)
    assert (
        len(per_label) == 3
    ), "Crawl error: wrong number of species, expecting 3 got %d".format(len(per_label))
    assert (
        img_cnt == 37
    ), "Download error: wrong total number of images, expecting 37 got %d".format(
        img_cnt
    )
    assert (
        val_cnt == 3
    ), "Split error: wrong number of validation images, expecting 3 got %d".format(
        val_cnt
    )


if __name__ == "__main__":
    test_script_execution()
