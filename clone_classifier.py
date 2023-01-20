"""
Original work: https://github.com/sangHa0411/CloneDetection
Copyright (c) 2022 Sangha Park(sangha110495), Young Jin Ahn(snoop2head)

Modified by Zihao Li
"""

import torch
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from datasets import Dataset

from utils.encoder import Encoder
from utils.trainer import ImprovedRDropTrainer
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import AnnotationPreprocessor, FunctionPreprocessor

from transformers import AutoTokenizer, AutoModel, TrainingArguments


class CloneClassifier:
    """
    A class that integrates data preprocessing, input tokenization, and model
    inferencing. It takes in a pandas dataframe with two columns: "code1" and
    "code2", and returns the predictions as a dataframe.
    """

    PLM = "Lazyhope/python-clone-detection"  # HuggingFace model name

    def __init__(self, args: argparse.Namespace = None):
        # -- Default training arguments
        if args is None:
            args = argparse.Namespace(
                max_token_size=512,
                fp16=False,
                input="",
                output_dir="results/",
                per_device_eval_batch_size=32,
            )

        self.training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
            remove_unused_columns=False,
        )

        # -- Tokenizing & Encoding
        self.tokenizer = AutoTokenizer.from_pretrained(self.PLM)
        self.encoder = Encoder(self.tokenizer, max_input_length=args.max_token_size)

        # -- Collator
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, max_length=args.max_token_size
        )

        # -- Config & Model
        self.model = AutoModel.from_pretrained(self.PLM, trust_remote_code=True)
        self.trainer_init()

    def trainer_init(self):
        """Initialize trainer."""
        self.trainer = ImprovedRDropTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
        )

    def enable_fp16(self):
        """Enable fp16 for faster inference if available."""
        self.training_args.fp16 = True
        self.trainer_init()
        print("[+] Fp16 enabled.")

    def prepare_inputs(self, df: pd.DataFrame):
        """Data preprocessing and tokenization."""
        # -- Loading datasets
        dset = Dataset.from_pandas(df)

        # -- Preprocessing datasets
        CPU_COUNT = multiprocessing.cpu_count() // 2

        fn_preprocessor = FunctionPreprocessor()
        dset = dset.map(fn_preprocessor, batched=True, num_proc=CPU_COUNT)

        an_preprocessor = AnnotationPreprocessor()
        dset = dset.map(an_preprocessor, batched=True, num_proc=CPU_COUNT)

        dset = dset.map(
            self.encoder,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=dset.column_names,
        )

        return dset

    def predict(
        self, df: pd.DataFrame, save_filename: str = None, return_score: bool = False
    ):
        """Perform model inference and return predictions as a dataframe."""
        # -- Preparing inputs
        dset = self.prepare_inputs(df)

        # -- Inference
        outputs = self.trainer.predict(dset)[0]  # logits output
        scores = torch.Tensor(outputs).softmax(dim=-1).numpy()  # probability output

        results = df[["code1", "code2"]].copy()
        results["predictions"] = np.argmax(scores, axis=-1)
        if return_score:
            # score of positive class
            if scores.size == 1:
                results["score"] = scores
            else:
                results["score"] = scores[:, 1]

        if save_filename is not None:
            path = Path(self.trainer.args.output_dir) / save_filename
            results.to_csv(path, index=False)

        return results
