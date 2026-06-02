"""BGE-M3 文本向量化 + CLIP 图片向量化"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Union
from loguru import logger


class Embedder:
    def __init__(self, use_bge: bool = True, use_clip: bool = True):
        self._bge = None
        self._clip_model = None
        self._clip_preprocess = None
        self._use_bge = use_bge
        self._use_clip = use_clip

    def _load_bge(self):
        if self._bge is None:
            from FlagEmbedding import BGEM3FlagModel
            from config import BGE_MODEL
            logger.info(f"Loading BGE model: {BGE_MODEL}")
            self._bge = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

    def _load_clip(self):
        if self._clip_model is None:
            import open_clip
            from config import CLIP_MODEL, CLIP_PRETRAIN
            logger.info(f"Loading CLIP model: {CLIP_MODEL}")
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL, pretrained=CLIP_PRETRAIN
            )
            self._clip_model.eval()
            self._clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    def encode_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load_bge()
        result = self._bge.encode(texts, batch_size=batch_size, max_length=256)
        return result["dense_vecs"].astype(np.float32)

    def encode_images(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> np.ndarray:
        self._load_clip()
        from PIL import Image
        all_vecs = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            images = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(self._clip_preprocess(img))
                except Exception:
                    images.append(torch.zeros(3, 224, 224))
            image_tensor = torch.stack(images)
            with torch.no_grad():
                vecs = self._clip_model.encode_image(image_tensor)
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)
            all_vecs.append(vecs.cpu().numpy())
        return np.vstack(all_vecs).astype(np.float32)

    def encode_image_from_pil(self, pil_images) -> np.ndarray:
        """Encode PIL images directly (for demo uploads)"""
        self._load_clip()
        import torch
        images = [self._clip_preprocess(img) for img in pil_images]
        image_tensor = torch.stack(images)
        with torch.no_grad():
            vecs = self._clip_model.encode_image(image_tensor)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy().astype(np.float32)

    def encode_text_clip(self, texts: List[str]) -> np.ndarray:
        """CLIP text encoder for image-text joint retrieval"""
        self._load_clip()
        tokens = self._clip_tokenizer(texts)
        with torch.no_grad():
            vecs = self._clip_model.encode_text(tokens)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy().astype(np.float32)
