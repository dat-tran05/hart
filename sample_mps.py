import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from hart.modules.models.transformer import HARTForT2I
from hart.utils import encode_prompts, llm_system_prompt


def save_images(sample_imgs, output):
    sample_img_np = sample_imgs[0].float().clamp(0, 1).mul(255).cpu().numpy()
    sample_img_np = sample_img_np.transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    Image.fromarray(sample_img_np).save(output)
    print(f"Image saved to {output}")


def main(args):
    device = torch.device("mps")
    dtype = torch.float16 if args.precision == "float16" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = AutoModel.from_pretrained(args.model_path)
    if args.use_ema:
        model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"), map_location="cpu", weights_only=False)
        )
    model = model.to(device=device, dtype=dtype)
    model.vae_proxy[0].to(device=device, dtype=dtype)  # VAE is in a tuple
    model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device=device, dtype=dtype)
    text_model.eval()

    prompts = [args.prompt]

    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast("mps", enabled=dtype == torch.float16, dtype=torch.float16):
            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                True,
            )

            output_imgs = model.autoregressive_infer_cfg(
                B=context_tensor.size(0),
                label_B=context_tensor.to(dtype),
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )
        torch.mps.synchronize()

    total_time = time.time() - start_time
    print(f"Generate {len(prompts)} images take {total_time:2f}s.")
    save_images(output_imgs, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="sample_mps.png")
    parser.add_argument("--precision", choices=("float32", "float16"), default="float16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--more_smooth", action="store_true")
    args = parser.parse_args()

    main(args)
