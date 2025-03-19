import fire
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing

from optimus.trainer.configuration.configs import Config
from optimus.trainer.data import Data, patch_spanner
from optimus.trainer.distributed import Distributed
from optimus.trainer.model.load import load_model, load_tokenizer
from optimus.trainer.pretrain import Pretrain


@torch.distributed.elastic.multiprocessing.errors.record
def main(**kwargs):
    # Load configurations
    config = Config(**kwargs)

    # Distributed training setup
    distributed = None
    if config.use_ddp or config.use_fsdp:
        distributed = Distributed(config)

    # Load/set model and get tokenizer.
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    if distributed:
        dist.barrier()
        config.log_print("Model and tokenizer loaded on all rank.")

    # Setup model for distributed training
    if config.use_fsdp:
        config.log_print("Shared model training with FSDP.")
        model = distributed.fsdp_setup_model(model)
    elif config.use_ddp:
        config.log_print("Distributed model training with DDP.")
        model = distributed.ddp_setup_model(model)

    patch_spanner()
    if distributed:
        dist.barrier()
    config.log_print("Mosaic ML Streaming spanner patched successfully.")

    # Load data
    data = Data(config, tokenizer)
    if distributed:
        dist.barrier()
    config.log_print("Data loaded.")

    # Train model
    pretrain = Pretrain(model, data, distributed, config)
    pretrain.train()

    # Cleanup distributed training
    if distributed:
        distributed.cleanup()
    config.log_print("Training completed successfully.")

    exit(0)


if __name__ == "__main__":
    fire.Fire(main)
