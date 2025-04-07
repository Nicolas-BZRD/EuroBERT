import torch

from optimus.trainer.model import model

try:
    from liger_kernel.nn import LigerRMSNorm

    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False


eurobert_config = {
    "210m": {
        "vocab_size": 128_256,
        "embedding_size": 768,
        "num_head": 12,
        "num_kv_head": 12,
        "num_layer": 12,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 3072,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "attn_impl": "torch",
        "rope_base": 10_000,
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    },
    "610m": {
        "vocab_size": 128_256,
        "embedding_size": 1152,
        "num_head": 18,
        "num_kv_head": 6,
        "num_layer": 26,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 4096,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "attn_impl": "torch",
        "rope_base": 10_000,
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    },
    "1b": {
        "vocab_size": 128_256,
        "embedding_size": 1152,
        "num_head": 18,
        "num_kv_head": 6,
        "num_layer": 26,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 4096,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "attn_impl": "torch",
        "rope_base": 10_000,
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    },
    "2b": {
        "vocab_size": 128_256,
        "embedding_size": 2304,
        "num_head": 18,
        "num_kv_head": 6,
        "num_layer": 32,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 6144,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "rope_base": 10_000,
        "attn_impl": "torch",
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    },
}


class EuroBERT(model.TransformerEncoder):
    def __init__(self, config):
        head_dim = config["embedding_size"] // config["num_head"]
        if config["fused_rms_norm"]:
            assert (
                LIGER_KERNEL_AVAILABLE
            ), "Liger kernel is not available. Please install it to use fused RMSNorm."
            norm = LigerRMSNorm
        else:
            norm = model.RMSNorm
        super().__init__(
            embedding=model.CustomEmbedding(
                config["vocab_size"], config["embedding_size"]
            ),
            blocks=[
                model.Block(
                    attention=model.SelfAttention(
                        embed_dim=config["embedding_size"],
                        head_dim=head_dim,
                        num_heads=config["num_head"],
                        num_kv_heads=config["num_kv_head"],
                        dropout=config["dropout"],
                        block_size=config["block_size"],
                        rope=model.RoPE(
                            dim=head_dim,
                            block_size=config["block_size"],
                            base=config["rope_base"],
                            fused_rope=config["fused_rope"],
                        ),
                        bias=config["bias"],
                        flash=config["attn_impl"] == "flash",
                    ),
                    mlp=model.SwigluMLP(
                        embed_dim=config["embedding_size"],
                        hidden_dim=config["mlp_hidden_dim"],
                        dropout=config["dropout"],
                        bias=config["bias"],
                        fused_swiglu=config["fused_swiglu"],
                    ),
                    attn_norm=norm(
                        config["embedding_size"],
                        eps=config["rms_norm_eps"],
                    ),
                    mlp_norm=norm(
                        config["embedding_size"],
                        eps=config["rms_norm_eps"],
                    ),
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layer"])
            ],
            final_layernorm=model.RMSNorm(
                config["embedding_size"],
                eps=config["rms_norm_eps"],
            ),
            lm_head=torch.nn.Linear(
                config["embedding_size"], config["vocab_size"], bias=config["bias"]
            ),
            fused_cross_entropy=config["fused_cross_entropy"],
        )
        # Tie the weights of the linear layer and the embedding layer
        if config["tied_weights"]:
            self.lm_head.weight = self.embedding.weight

    @property
    def device(self):
        return next(self.parameters()).device
