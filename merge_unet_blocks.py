# For example:
# python tools/merge_unet_blocks.py xxx1.safetensors xxx2.safetensors \
#         --input_blocks "0:0.5, 1:0.5, 2:0.5, 3:0.6, 4:0.5, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.5" \
#         --middle_blocks "0:0.5, 1:0.5, 2:0.6" \
#         --output_blocks "0:0.5, 1:0.5, 2:0.5, 3:0.6, 4:0.5, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.5" \
#         --out "0:0.5, 2:0.3" \
#         --time_embed "0:0.5, 2:0.3" \
#         --dump_path ./merged.safetensors
# or (same as above):
# python tools/merge_unet_blocks.py xxx1.safetensors xxx2.ckpt \
#         --base_alpha 0.5 \
#         --input_blocks "3:0.6" \
#         --middle_blocks "2:0.6" \
#         --output_blocks "3:0.6," \
#         --out "2:0.3" \
#         --time_embed "2:0.3" \
#         --dump_path ./merged.safetensors
# or just (merge all blocks with base_alpha):
# python tools/merge_unet_blocks.py xxx1.ckpt xxx2.ckpt \
#         --base_alpha 0.5 \
#         --dump_path ./merged.ckpt
from safetensors.torch import load_file, save_file
import torch
import argparse, os, re


def load_weights(path):
    if path.endswith(".safetensors"):
        weights = load_file(path, "cpu")
    else:
        weights = torch.load(path, map_location="cpu")
        weights = weights["state_dict"] if "state_dict" in weights else weights

    return weights


def save_weights(weights, path):
    if path.endswith(".safetensors"):
        save_file(weights, path)
    else:
        torch.save({"state_dict": weights}, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model1", type=str, help="Path to the input file.")
    parser.add_argument("model2", type=str, help="Path to the output file.")
    parser.add_argument(
        "--dump_path", type=str, default=None, help="Path to the output file."
    )
    parser.add_argument(
        "--base_alpha", type=float, default=0, help="Base alpha value for model2."
    )
    parser.add_argument(
        "--input_blocks",
        type=str,
        default=None,
        help="Input blocks index and alpha for merge.",
    )
    parser.add_argument(
        "--middle_blocks",
        type=str,
        default=None,
        help="Middle blocks index and alpha for merge.",
    )
    parser.add_argument(
        "--output_blocks",
        type=str,
        default=None,
        help="Output blocks index and alpha for merge.",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Out blocks index and alpha for merge."
    )
    parser.add_argument(
        "--time_embed",
        type=str,
        default=None,
        help="Time embedding blocks index and alpha for merge.",
    )
    args = parser.parse_args()

    weights1 = load_weights(args.model1)
    weights2 = load_weights(args.model2)

    merge_keys = []
    merge_alphas = []

    if args.input_blocks is not None:
        for i in re.findall(r"(\d+):(\d+\.?\d*)", args.input_blocks):
            index, alpha = i
            merge_keys.append(f"input_blocks.{index}")
            merge_alphas.append(float(alpha))

    if args.middle_blocks is not None:
        for i in re.findall(r"(\d+):(\d+\.?\d*)", args.middle_blocks):
            index, alpha = i
            merge_keys.append(f"middle_block.{index}")
            merge_alphas.append(float(alpha))

    if args.output_blocks is not None:
        for i in re.findall(r"(\d+):(\d+\.?\d*)", args.output_blocks):
            index, alpha = i
            merge_keys.append(f"output_blocks.{index}")
            merge_alphas.append(float(alpha))

    if args.out is not None:
        for i in re.findall(r"(\d+):(\d+\.?\d*)", args.out):
            index, alpha = i
            merge_keys.append(f"out.{index}")
            merge_alphas.append(float(alpha))

    if args.time_embed is not None:
        for i in re.findall(r"(\d+):(\d+\.?\d*)", args.time_embed):
            index, alpha = i
            merge_keys.append(f"time_embed.{index}")
            merge_alphas.append(float(alpha))

    # Merge Unet blocks' weights
    for k in weights1.keys():
        if "diffusion_model" in k and k in weights2:
            for index, merge_key in enumerate(merge_keys):
                if merge_key in k:
                    print(
                        f"{k} = model1 * {1 - merge_alphas[index]} + model2 * {merge_alphas[index]}"
                    )
                    weights1[k] = (
                        weights1[k] * (1 - merge_alphas[index])
                        + weights2[k] * merge_alphas[index]
                    )
                    break
            else:
                print(
                    f"{k} = model1 * {1 - args.base_alpha} + model2 * {args.base_alpha}"
                )
                weights1[k] = (
                    weights1[k] * (1 - args.base_alpha) + weights2[k] * args.base_alpha
                )

    if args.dump_path is not None:
        dump_path = args.dump_path

    else:
        weights1_name = os.path.basename(args.model1)
        weights2_name = os.path.basename(args.model2)
        merged_name = f"{weights1_name}_{weights2_name}.safetensors"
        dump_path = os.path.join(os.path.dirname(args.model1), merged_name)

    print(f"Save merged weights to {dump_path}")
    save_weights(weights1, dump_path)

    print("Done!")
