"""Reusable VLM client abstraction with caching and retry handling.

To add a new VLM client:
1. Implement BaseVLMClient._answer_uncached(image_path, question).
2. Register the class in CLIENT_REGISTRY.
3. Add a model entry in configs/models.yaml or pass a custom config with --config.

Evaluation code should depend only on client.answer(image_path, question).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from .cache import ResponseCache
except ImportError:  # Allows `python src/evaluate.py` from the repo root.
    from cache import ResponseCache


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_CONFIG_PATH = ROOT_DIR / "configs" / "models.yaml"
DEFAULT_CACHE_PATH = ROOT_DIR / "results" / "cache" / "model_response_cache.jsonl"
MAX_RETRIES = 2


class VLMClientError(RuntimeError):
    """Raised when a VLM client cannot answer safely."""


class BaseVLMClient:
    """Shared interface for all VLM clients."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        self.model_name = model_name
        self.config = config
        self.cache = cache or ResponseCache(DEFAULT_CACHE_PATH)
        self.max_retries = int(config.get("max_retries", str(MAX_RETRIES)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"

    def answer(self, image_path: str, question: str, perturbation: str | None = None) -> str:
        """Return an answer for one image/question pair using a shared cache."""
        cached = self.cache.get(self.model_name, image_path, question, perturbation)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                answer = self._answer_uncached(image_path, question)
                self.cache.set(self.model_name, image_path, question, answer, perturbation)
                return answer
            except Exception as exc:  # noqa: BLE001 - converted to a clear framework error.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.1)
        raise VLMClientError(f"{self.model_name} failed after {self.max_retries} attempt(s): {last_error}")

    def _answer_uncached(self, image_path: str, question: str) -> str:
        raise NotImplementedError


class MockVLMClient(BaseVLMClient):
    """Deterministic local client for dry runs and tests."""

    def _answer_uncached(self, image_path: str, question: str) -> str:
        normalized = question.lower()
        if "total" in normalized:
            return "mock_total"
        if "status" in normalized:
            return "mock_status"
        if "highest" in normalized:
            return "mock_highest"
        if "review days" in normalized:
            return "mock_days"
        return "mock_answer"


class OpenAIStubClient(BaseVLMClient):
    """API client stub for future OpenAI-compatible VLM integration."""

    def _answer_uncached(self, image_path: str, question: str) -> str:
        env_key = self.config.get("api_key_env", "OPENAI_API_KEY")
        if not os.getenv(env_key):
            raise VLMClientError(
                f"{self.model_name} requires {env_key}. Set it in the environment; do not commit API keys."
            )
        raise VLMClientError("OpenAI VLM API call is not implemented in Day 3; add it behind this interface later.")


def _bool_config(config: dict[str, str], key: str, default: bool = False) -> bool:
    value = str(config.get(key, str(default))).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _int_config(config: dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(config.get(key, str(default))).strip())
    except ValueError:
        return default


def _resolved_image_path(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise VLMClientError(f"image path does not exist: {path}")
    return str(path.resolve())


def _resolved_image_uri(image_path: str) -> str:
    return Path(_resolved_image_path(image_path)).as_uri()


def _prompt_for_model(config: dict[str, str], question: str) -> str:
    suffix = str(config.get("prompt_suffix", "")).strip()
    if not suffix:
        return question
    return f"{question.strip()}\n{suffix}"


def _clean_generated_answer(text: str) -> str:
    answer = str(text).strip()
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"^```(?:text)?|```$", "", answer, flags=re.IGNORECASE | re.MULTILINE).strip()
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return answer

    has_protocol = any(re.match(r"^(?:final\s+answer|answer|evidence|rationale|reason)\s*:", line, re.IGNORECASE) for line in lines)
    if has_protocol and len(lines) > 1:
        return "\n".join(lines)

    single = lines[0]
    single = re.sub(r"^<answer>", "", single, flags=re.IGNORECASE).strip()
    single = re.sub(r"</answer>$", "", single, flags=re.IGNORECASE).strip()
    for prefix in ("Final answer:", "final answer:", "Answer:", "answer:"):
        if single.startswith(prefix):
            return single[len(prefix) :].strip()
    return "\n".join(lines)


def _cuda_or_model_device(model: Any, torch_module: Any) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


def _model_load_kwargs(config: dict[str, str], torch_module: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"device_map": config.get("device_map", "auto")}
    if _bool_config(config, "load_in_4bit", False):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise VLMClientError("load_in_4bit requires bitsandbytes and recent transformers") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        dtype = config.get("torch_dtype", "auto")
        if dtype == "float16":
            kwargs["torch_dtype"] = torch_module.float16
        elif dtype == "bfloat16":
            kwargs["torch_dtype"] = torch_module.bfloat16
        else:
            kwargs["torch_dtype"] = dtype
    if "attn_implementation" in config:
        kwargs["attn_implementation"] = config["attn_implementation"]
    if "low_cpu_mem_usage" in config:
        kwargs["low_cpu_mem_usage"] = _bool_config(config, "low_cpu_mem_usage", True)
    return kwargs


class HFChatTemplateClient(BaseVLMClient):
    """Generic Hugging Face chat-template client for recent multimodal models."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        super().__init__(model_name, config, cache)
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise VLMClientError(
                "HFChatTemplateClient requires torch, transformers, accelerate, pillow, and optional bitsandbytes."
            ) from exc

        self.torch = torch
        self.model_id = config["model_id"]
        self.max_new_tokens = _int_config(config, "max_new_tokens", 64)
        model_class = config.get("model_class", "AutoModelForMultimodalLM")
        try:
            transformers_module = __import__("transformers", fromlist=[model_class])
            model_cls = getattr(transformers_module, model_class)
        except (ImportError, AttributeError) as exc:
            raise VLMClientError(
                f"{model_class} is unavailable. Update transformers in the Kaggle notebook before using {self.model_id}."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, use_fast=True)
        self.model = model_cls.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **_model_load_kwargs(config, torch),
        )

    def _answer_uncached(self, image_path: str, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        self.config.get("image_content_key", "image"): _resolved_image_path(image_path),
                    },
                    {"type": "text", "text": _prompt_for_model(self.config, question)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_cuda_or_model_device(self.model, self.torch))
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _clean_generated_answer(decoded)


class Qwen25VLClient(BaseVLMClient):
    """Optional local Hugging Face client for Qwen2.5-VL on Kaggle/GPU runtimes."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        super().__init__(model_name, config, cache)
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise VLMClientError(
                "Qwen25VLClient requires torch, transformers, accelerate, qwen-vl-utils, and pillow. "
                "Install the optional Kaggle dependencies before using this client."
            ) from exc

        self.torch = torch
        self.model_id = config.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.max_new_tokens = _int_config(config, "max_new_tokens", 64)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **_model_load_kwargs(config, torch),
        )

    def _answer_uncached(self, image_path: str, question: str) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise VLMClientError("Qwen25VLClient requires qwen-vl-utils") from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _resolved_image_path(image_path)},
                    {"type": "text", "text": _prompt_for_model(self.config, question)},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_cuda_or_model_device(self.model, self.torch))
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return _clean_generated_answer(self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


class SmolVLMClient(BaseVLMClient):
    """Optional local Hugging Face client for SmolVLM2 image-text generation."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        super().__init__(model_name, config, cache)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise VLMClientError(
                "SmolVLMClient requires torch, transformers, accelerate, pillow, and optional bitsandbytes."
            ) from exc

        self.torch = torch
        self.model_id = config.get("model_id", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        self.max_new_tokens = _int_config(config, "max_new_tokens", 64)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **_model_load_kwargs(config, torch),
        )

    def _answer_uncached(self, image_path: str, question: str) -> str:
        from PIL import Image

        image = Image.open(_resolved_image_path(image_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _prompt_for_model(self.config, question)},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(
            _cuda_or_model_device(self.model, self.torch)
        )
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        return _clean_generated_answer(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0])


class InternVL3Client(BaseVLMClient):
    """Optional local Hugging Face client for InternVL3 image-question answering."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        super().__init__(model_name, config, cache)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise VLMClientError(
                "InternVL3Client requires torch, transformers, accelerate, pillow, torchvision, timm, and einops."
            ) from exc

        self.torch = torch
        self.model_id = config.get("model_id", "OpenGVLab/InternVL3-2B")
        self.max_new_tokens = _int_config(config, "max_new_tokens", 64)
        self.input_size = _int_config(config, "input_size", 448)
        self.max_tiles = _int_config(config, "max_tiles", 4)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        load_kwargs = _model_load_kwargs(config, torch)
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, **load_kwargs).eval()
        if not _bool_config(config, "load_in_4bit", False) and torch.cuda.is_available():
            self.model = self.model.cuda()

    @staticmethod
    def _build_transform(input_size: int) -> Any:
        try:
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:
            raise VLMClientError("InternVL3Client requires torchvision") from exc

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def _closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: set[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
        return best_ratio

    @classmethod
    def _dynamic_preprocess(cls, image: Any, image_size: int, max_num: int) -> list[Any]:
        width, height = image.size
        aspect_ratio = width / height
        target_ratios = {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num
        }
        target_aspect_ratio = cls._closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized = image.resize((target_width, target_height))
        processed = []
        for index in range(blocks):
            box = (
                (index % (target_width // image_size)) * image_size,
                (index // (target_width // image_size)) * image_size,
                ((index % (target_width // image_size)) + 1) * image_size,
                ((index // (target_width // image_size)) + 1) * image_size,
            )
            processed.append(resized.crop(box))
        return processed

    def _answer_uncached(self, image_path: str, question: str) -> str:
        from PIL import Image

        image = Image.open(_resolved_image_path(image_path)).convert("RGB")
        transform = self._build_transform(self.input_size)
        images = self._dynamic_preprocess(image, image_size=self.input_size, max_num=self.max_tiles)
        pixel_values = self.torch.stack([transform(tile) for tile in images])
        dtype = self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32
        pixel_values = pixel_values.to(dtype=dtype, device=_cuda_or_model_device(self.model, self.torch))
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        prompt = f"<image>\n{_prompt_for_model(self.config, question)}"
        with self.torch.inference_mode():
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        return _clean_generated_answer(str(response))


class LlavaHFClient(BaseVLMClient):
    """Optional local Hugging Face client for llava-hf/llava-1.5 style models."""

    def __init__(self, model_name: str, config: dict[str, str], cache: ResponseCache | None = None) -> None:
        super().__init__(model_name, config, cache)
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:
            raise VLMClientError(
                "LlavaHFClient requires torch, transformers, accelerate, pillow, and optional bitsandbytes."
            ) from exc

        self.torch = torch
        self.model_id = config.get("model_id", "llava-hf/llava-1.5-7b-hf")
        self.max_new_tokens = _int_config(config, "max_new_tokens", 64)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **_model_load_kwargs(config, torch),
        )

    def _answer_uncached(self, image_path: str, question: str) -> str:
        from PIL import Image

        image = Image.open(_resolved_image_path(image_path)).convert("RGB")
        prompt = f"USER: <image>\n{_prompt_for_model(self.config, question)}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            _cuda_or_model_device(self.model, self.torch)
        )
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        return _clean_generated_answer(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0])


CLIENT_REGISTRY = {
    "mock": MockVLMClient,
    "openai_stub": OpenAIStubClient,
    "hf_chat_template": HFChatTemplateClient,
    "qwen25_vl": Qwen25VLClient,
    "smolvlm": SmolVLMClient,
    "internvl3": InternVL3Client,
    "llava_hf": LlavaHFClient,
}


def load_model_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, dict[str, str]]:
    """Load the small Day 3 models.yaml format without external YAML dependencies."""
    models: dict[str, dict[str, str]] = {}
    current_name: str | None = None
    if not path.exists():
        return {"mock": {"provider": "mock", "max_retries": str(MAX_RETRIES)}}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "models:":
            continue
        if stripped.startswith("- name:"):
            current_name = stripped.split(":", 1)[1].strip().strip("\"'")
            models[current_name] = {}
            continue
        if current_name and ":" in stripped:
            key, value = stripped.split(":", 1)
            models[current_name][key.strip()] = value.strip().strip("\"'")

    if "mock" not in models:
        models["mock"] = {"provider": "mock", "max_retries": str(MAX_RETRIES)}
    return models


def get_client(model_name: str, config_path: Path = MODELS_CONFIG_PATH) -> BaseVLMClient:
    """Construct a configured client by model name."""
    configs = load_model_config(config_path)
    if model_name not in configs:
        available = ", ".join(sorted(configs))
        raise VLMClientError(f"unknown model {model_name!r}; available models: {available}")
    config = configs[model_name]
    provider = config.get("provider", model_name)
    client_cls = CLIENT_REGISTRY.get(provider)
    if client_cls is None:
        available = ", ".join(sorted(CLIENT_REGISTRY))
        raise VLMClientError(f"provider {provider!r} is not registered; available providers: {available}")
    return client_cls(model_name=model_name, config=config)

