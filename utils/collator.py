# https://github.com/sangHa0411/CloneDetection/blob/main/utils/collator.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy


@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_ids2 = (
            [feature["input_ids2"] for feature in features]
            if "input_ids2" in features[0].keys()
            else None
        )
        padding_side = self.tokenizer.padding_side

        if input_ids2 is not None:
            max_input_ids2_length = max(len(l) for l in input_ids2)
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_input_ids2_length - len(feature["input_ids2"])
                )
                feature["input_ids2"] = (
                    feature["input_ids2"] + remainder
                    if padding_side == "right"
                    else remainder + feature["input_ids2"]
                )

        attention_mask2 = (
            [feature["attention_mask2"] for feature in features]
            if "attention_mask2" in features[0].keys()
            else None
        )
        if attention_mask2 is not None:
            max_attention_mask2_length = max(len(l) for l in attention_mask2)
            for feature in features:
                remainder = [0] * (
                    max_attention_mask2_length - len(feature["attention_mask2"])
                )
                feature["attention_mask2"] = (
                    feature["attention_mask2"] + remainder
                    if padding_side == "right"
                    else remainder + feature["attention_mask2"]
                )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
