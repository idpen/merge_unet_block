# merge_unet_block
a python script to merge stable diffusion's Unet blocks.

## For example:
```python
python merge_unet_blocks.py xxx1.safetensors xxx2.safetensors \
        --input_blocks "0:0.5, 1:0.5, 2:0.5, 3:0.6, 4:0.5, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.5" \
        --middle_blocks "0:0.5, 1:0.5, 2:0.6" \
        --output_blocks "0:0.5, 1:0.5, 2:0.5, 3:0.6, 4:0.5, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.5" \
        --out "0:0.5, 2:0.3" \
        --time_embed "0:0.5, 2:0.3" \
        --dump_path ./merged.safetensors
```
## or (same as above):
```python
python merge_unet_blocks.py xxx1.safetensors xxx2.ckpt \
        --base_alpha 0.5 \
        --input_blocks "3:0.6" \
        --middle_blocks "2:0.6" \
        --output_blocks "3:0.6," \
        --out "2:0.3" \
        --time_embed "2:0.3" \
        --dump_path ./merged.safetensors
```
## or just (merge all blocks with base_alpha):
```python
python merge_unet_blocks.py xxx1.ckpt xxx2.ckpt \
        --base_alpha 0.5 \
        --dump_path ./merged.ckpt
```
