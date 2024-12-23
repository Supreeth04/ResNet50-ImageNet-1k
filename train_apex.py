from apex import amp
from apex.parallel import DistributedDataParallel

def train_with_apex(model, optimizer):
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2",
        keep_batchnorm_fp32=True,
        loss_scale="dynamic"
    )
    model = DistributedDataParallel(model) 