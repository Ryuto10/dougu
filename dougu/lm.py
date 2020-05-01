# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from torch.distributions import Categorical
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


class BertMlmInteractive:
    """
    Args:
        how_pickup: choose from ['argmax', 'sample'].
            - 'argmax': Select the most probable token
            - 'sample': Select the token by sampling
        what_order: choose from ['beam', 'multi', 'single']
            - 'beam': Search for tokens with beam search
            - 'multi': Fill the MASK with tokens at the same time
            - 'single': Recursively fill MASK from the front with tokens
    Setting:
        self.how_pickup: str
        self.whar_order: str
        self.beam_k: int
        self.black_list: [str, ...]
    """
    def __init__(self, how_pickup: str = "argmax", what_order: str = "beam"):
        # setting
        self.how_pickup: str = how_pickup
        self.what_order: str = what_order
        self.beam_k: int = 5
        self.black_list: [str, ...] = []

        self._black_list_ids = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Device: {}".format(self._device))
        logger.debug("Loading bert tokenizer...")
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
        logger.debug("Loading bert model...")
        self.model = BertForMaskedLM.from_pretrained('bert-base-japanese-whole-word-masking')
        self.model.to(self._device)
        self.model.eval()

    def __call__(self, text):
        input_ids = self._convert_text_to_ids(text)
        masked_ids = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()
        if not masked_ids:
            logger.warning("Can't find mask token. A mask token is denoted by 'M'.")
            return None
        self.black_list = [token for token in self.black_list if token in self.tokenizer.vocab]
        self._black_list_ids = self.tokenizer.convert_tokens_to_ids(self.black_list) if self.black_list else []
        with torch.no_grad():
            if self.what_order == "beam":
                predict_topk_sents, predict_topk_tokens = self._prediction_with_beam_search(input_ids, masked_ids)
                return predict_topk_sents
            elif self.what_order == "multi":
                predict_tokens = self._prediction_multi(input_ids, masked_ids)
            elif self.what_order == "single":
                predict_tokens = self._prediction_single(input_ids, masked_ids)
            else:
                raise ValueError("Unsupported value: {}".format(self.what_order))
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        for idx, token in zip(masked_ids, predict_tokens):
            input_tokens[idx] = token

        return input_tokens

    def _convert_text_to_ids(self, text) -> torch.Tensor:
        text = text.replace("M", self.tokenizer.mask_token)
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self._device)  # shape = (1, seq_length)

        return input_ids

    def _prediction_with_beam_search(self, input_ids: torch.Tensor, masked_ids: [int, ...]):
        topk_dict = {0: (1, [])}
        for n, masked_id in enumerate(masked_ids):
            topk_buffer = []
            for prob, predicted_ids in topk_dict.values():
                if n == 0:
                    predict = self.model(input_ids)[0]
                else:
                    filled_ids = input_ids.clone()
                    filled_ids[0][masked_ids[:n]] = torch.LongTensor(predicted_ids).to(self._device)
                    predict = self.model(filled_ids)[0]
                predict[:, masked_id, self._black_list_ids] = -np.inf
                predict = torch.softmax(predict, dim=2)
                topk_prob, topk_indices = predict.topk(self.beam_k)
                topk_buffer += [(prob * float(p), predicted_ids + [int(idx)])
                                for p, idx in zip(topk_prob[0][masked_id], topk_indices[0][masked_id])]
            topk_dict = {i: p_ids for i, p_ids in enumerate(sorted(topk_buffer, key=lambda x: -x[0])[:self.beam_k])}

        predict_topk_sents = []
        predict_topk_tokens = []
        for i in range(self.beam_k):
            output_ids = input_ids[0].tolist()
            for idx, token in zip(masked_ids, topk_dict[i][1]):
                output_ids[idx] = token
            predict_topk_sents.append(self.tokenizer.convert_ids_to_tokens(output_ids))
            predict_topk_tokens.append(self.tokenizer.convert_ids_to_tokens(topk_dict[i][1]))

        return predict_topk_sents, predict_topk_tokens

    def _prediction_single(self, input_ids: torch.Tensor, masked_ids: [int, ...]):
        all_predict_ids = []
        for n, masked_id in enumerate(masked_ids):
            pred_ids = self._prediction(input_ids)
            all_predict_ids.append(int(pred_ids[0][masked_id]))
            input_ids[0][masked_id] = pred_ids[0][masked_id]
        predict_tokens = self.tokenizer.convert_ids_to_tokens(all_predict_ids)

        return predict_tokens

    def _prediction_multi(self, input_ids: torch.Tensor, masked_ids: [int, ...]):
        pred_ids = self._prediction(input_ids)
        all_predict_ids = [int(pred_ids[0][masked_id]) for masked_id in masked_ids]
        predict_tokens = self.tokenizer.convert_ids_to_tokens(all_predict_ids)

        return predict_tokens

    def _prediction(self, input_ids: torch.Tensor):
        predict = self.model(input_ids)[0]
        predict[:, :, self._black_list_ids] = -np.inf
        if self.how_pickup == "sample":
            dist = Categorical(logits=F.log_softmax(predict, dim=-1))
            pred_ids = dist.sample()
        elif self.how_pickup == "argmax":
            pred_ids = predict.argmax(dim=-1)
        else:
            raise ValueError("Selection mechanism %s not found!" % self.how_pickup)

        return pred_ids
