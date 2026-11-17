from enum import Enum
import torch
import torch.nn.functional as F

from api.schemas import GesturePrediction

class TaskType(str, Enum):
    CLASSIFICATION = "classification", # multi-class classification
    CTC = "ctc" # connectionist temporal classification

# Simple classification task
# logits to top-k
class MultiClassDecoder:
    """
    Decodes raw logits into a top-k prediction list.
    Converts a 1-D tensor of length NUM_CLASSES to a ranked list of GesturePrediction objects.
    """
    def __init__(self, class_map: dict[str, int]):
        self.class_map = {idx: name for name, idx in class_map.items()}

    def decode(self, logits: torch.Tensor, k: int = 3) -> list[GesturePrediction]:
        """
        logits: (NUM_CLASSES,) float32 - RAW scores (not softmaxed) for ONE clip.
        """
        n_classes = len(self.class_map)
        if logits.shape[-1] != n_classes:
            raise RuntimeError(
                f"ERROR: Model output has {logits.shape[-1]} logits but classes file defines {n_classes} classes. "
                f"Make sure the right class list and model weights are being used."
            )

        probs = F.softmax(logits, dim=-1)
        top_probs, top_idxs = torch.topk(probs, k=min(k, n_classes))
        return [
            GesturePrediction(
                label=self.class_map[int(idx.item())],
                class_index=int(idx.item()),
                confidence=round(prob.item(), 4)
            )
            for prob, idx in zip(top_probs, top_idxs)
        ]

# CTC task
class CTCGreedyDecoder:
    """
    Decodes raw logits into a list of token predictions using greedy CTC.
    Needs a "blank" index entry in the class list.
    """
    def __init__(self, class_map: dict[str, int], blank_index: int = 0):
        self.class_map = {idx: name for name, idx in class_map.items()}
        if blank_index not in self.class_map:
            raise ValueError(f"Blank index={blank_index} is not in the class map - add <blank> to the class json.")
        self.blank_index = blank_index

    def decode(self, log_probs: torch.Tensor, n_frames: int, offset: int = 0) -> list[GesturePrediction]:
        """
        `log_probs`: (T', C) - a single clip, batch-first, with log-softmax applied;
        `n_frames`: T timesteps to feed into the model;
        `offset`: first frame index in the source.
        """
        n_timesteps, n_outputs = log_probs.shape
        if n_outputs != len(self.class_map):
            raise RuntimeError(f"Model output has {n_outputs} classes but the classes file defines {len(self.class_map)}. It should include a <blank>.")
        
        frames_per_step = n_frames / n_timesteps
        step_probs, best_pred = log_probs.exp().max(dim=-1) # confidence, best label per t
        step_probs, best_pred = step_probs.tolist(), best_pred.tolist()

        tokens: list[GesturePrediction] = []
        previous, start = self.blank_index, 0

        # Trailing -1 is for the last run
        for t, i in enumerate(best_pred + [-1]):
            if i == previous:
                continue
            if previous != self.blank_index: # not a blank, append one token
                first = offset + int(round(start*frames_per_step))
                last = offset + int(round(t*frames_per_step)) - 1
                tokens.append(
                    GesturePrediction(
                        label=self.class_map[previous],
                        class_index=previous,
                        confidence=round(max(step_probs[start:t]), 4),
                        start_frame=first,
                        end_frame=max(first, last)
                    )
                )
            previous, start = i, t
        return tokens

    @staticmethod
    def to_text(tokens: list[GesturePrediction], joiner: str = "") -> str:
        """Predicted letters join without a separator."""
        return joiner.join(t.label for t in tokens)
