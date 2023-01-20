# https://github.com/sangHa0411/CloneDetection/blob/main/utils/encoder.py
from transformers.tokenization_utils_base import BatchEncoding


class Encoder:
    def __init__(self, tokenizer, max_input_length: int):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __call__(self, examples):
        code1_inputs = self.tokenizer(
            examples["code1"],
            examples["code2"],
            max_length=self.max_input_length,
            return_token_type_ids=False,
            truncation=True,
        )

        code2_inputs = self.tokenizer(
            examples["code2"],
            examples["code1"],
            max_length=self.max_input_length,
            return_token_type_ids=False,
            truncation=True,
        )

        model_inputs = BatchEncoding(
            {
                "input_ids": code1_inputs["input_ids"],
                "attention_mask": code1_inputs["attention_mask"],
                "input_ids2": code2_inputs["input_ids"],
                "attention_mask2": code2_inputs["attention_mask"],
            }
        )

        if "similar" in examples:
            model_inputs["labels"] = examples["similar"]

        return model_inputs
