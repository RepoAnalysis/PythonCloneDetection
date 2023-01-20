# PythonCloneDetection

Detect semantically similar python code using fine-tuned GraphCodeBERT model.

## About

This modified [GraphCodeBERT](https://arxiv.org/abs/2009.08366) model was fine-tuned for 11 hours using an A40 server on the [PoolC (1fold)](https://huggingface.co/datasets/PoolC/1-fold-clone-detection-600k-5fold) dataset, which contains over 6M pairs of semantically similar python code snippets.

It is then used to predict the similarity of python code snippets in other folds of the [PoolC](https://huggingface.co/datasets/PoolC/5-fold-clone-detection-600k-5fold) dataset, as well as the [C4](https://github.com/Chenning-Tao/C4/tree/main/dataset) dataset. It achieved F1 scores of greater than 0.97 on all datasets in several experiments, where balanced sampling was applied.

## Prerequisites & Installation

* pip

    In your virtual environment, run:

    ```sh
    pip install -r requirements.txt
    ```

    to install the required packages.

* conda

    To create a new conda environment called `PythonCloneDetection` with the required packages, run:

    ```sh
    conda env create -f environment.yml
    ```

    (this may take a while to finish)

The above commands will install cpu-only version of the `pytorch` package. Please refer to [PyTorch's official website](https://pytorch.org/get-started/locally/) for instructions on how to install other versions of `pytorch` on your machine.

## Usage

1. Run `python main.py --input <input_file> --output_dir <output_directory>` to run `CloneClassifier` on the specified input file and save the predictions as `results.csv` in the specified output directory. By default, the input file is `examples/c4.csv` and the output directory is `results/`.
2. Use the command `python main.py --help` to see other optional arguments including `max_token_size`, `fp16`, and `per_device_eval_batch_size`.
3. You could also import `CloneClassifier` class from `clone_classifier.py` and use it in your own code, for example:

    ```python
    import argparse
    import pandas as pd
    from clone_classifier import CloneClassifier


    args = argparse.Namespace(
                max_token_size=512,
                fp16=False,
                input="",
                output_dir="results/",
                per_device_eval_batch_size=8,
            )
    classifier = CloneClassifier(args)
    # enable fp16 for faster inference if available:
    # classifier.enable_fp16()

    df = pd.read_csv("examples/c4.csv").head(10)
    res_df = classifier.predict(df[["code1", "code2"]])

    print(res_df["predictions"] == df["similar"])
    # res_df.to_csv("results/results.csv", index=False)
    ```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [GraphCodeBERT](https://arxiv.org/abs/2009.08366)
* [Original work of the model by @snoop2head](https://github.com/sangHa0411/CloneDetection)
* [Dataset source from dacon](https://dacon.io/competitions/official/235900/overview/description)
* [Dataset shared by PoolC](https://huggingface.co/PoolC)
