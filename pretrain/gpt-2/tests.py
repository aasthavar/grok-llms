import torch
from transformer_lens import EasyTransformer

from model import *

def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    
    random_input = torch.randn(shape).cuda()
    print(f"input: {random_input.shape}")
    
    output = layer(random_input)
    print(f"output: {output.shape}")
    return output
    
    
def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    
    random_input = torch.randint(100, 1000, shape).cuda()
    print(f"input: {random_input.shape}")
    
    output = layer(random_input)
    print(f"output: {output.shape}")
    return output

def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(
        gpt2_layer.state_dict(),
        strict=False
    )
    
    if isinstance(input_name, str):
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    print(f"test | reference_input: {reference_input.shape}")
    
    output = layer(reference_input)
    print(f"test | output: {output.shape}")
    
    reference_output = gpt2_layer(reference_input)
    print(f"test | reference_output: {reference_output.shape}")
    
    comparison = torch.isclose(
        output, reference_output, atol=1e-4, rtol=1e-3
    )
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    print(f"-"*10)
    return output


if __name__ == "__main__":
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    reference_gpt2 = EasyTransformer.from_pretrained(
        "gpt2-small", 
        fold_ln=False, 
        center_unembed=False, 
        center_writing_weights=False
    )
    tokens = reference_gpt2.to_tokens(reference_text)
    logits, cache = reference_gpt2.run_with_cache(tokens)
    
    # print(f"testing LayerNorm ...")
    # _ = rand_float_test(LayerNorm, [2, 4, 768])
    # _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post", cache.cache_dict)
    
    # print(f"testing Embed ...")
    # _ = rand_int_test(Embed, [2, 4])
    # _  = load_gpt2_test(Embed, reference_gpt2.embed, tokens, cache.cache_dict)
    
    # print(f"testing PosEmbed ...")
    # _ = rand_int_test(PosEmbed, [2, 4])
    # _ = load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens, cache.cache_dict)
    
    print(f"testing Attention ...")
    _ = rand_float_test(Attention, [2, 4, 768])
    _ = load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"], cache.cache_dict)
    