# https://github.com/sangHa0411/CloneDetection/blob/main/trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ImprovedRDropTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                outputs1 = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                outputs2 = model(
                    input_ids=inputs["input_ids2"],
                    attention_mask=inputs["attention_mask2"],
                )

                logits = (outputs1.logits + outputs2.logits) / 2

        if prediction_loss_only:
            return (loss, None, None)

        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_eval_loss(self, model, inputs, return_outputs=False):
        num_labels = 2
        labels = inputs.pop("labels")

        outputs1 = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        outputs2 = model(
            input_ids=inputs["input_ids2"], attention_mask=inputs["attention_mask2"]
        )

        logits = (outputs1.logits + outputs2.logits) / 2

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return (loss, logits) if return_outputs else loss
