from normflow.mask import EvenOddMask
from normflow.nn import (
    AffineCoupling_,
    ConvAct,
    DistConvertor_,
    FFTNet_,
    Identity_,
    MeanFieldNet_,
    ModuleList_,
    PSDBlock_,
)


class ScalarPSDAffineCoupling(ModuleList_):
    def __init__(self, lat_shape, **kwargs):
        nets_list = assemble_net(lat_shape, **kwargs)
        super().__init__(nets_list)


# =============================================================================
def assemble_net(
    lat_shape,
    n_layers=4,
    hidden_sizes=[8, 8],
    zee2sym=True,
    acts=None,
    knots0_len=10,
    knots1_len=10,
    knots2_len=50,
    knots4_len=50,
):
    mfdict = dict(knots_len=knots0_len, symmetric=zee2sym, final_scale=True, smooth=True)

    fftdict = dict(knots_len=knots1_len, ignore_zeromode=True)

    nets_list = []
    # 1. First block
    mfnet_ = MeanFieldNet_.build(**mfdict) if (knots0_len > 1) else Identity_()
    fftnet_ = FFTNet_.build(lat_shape, **fftdict)
    nets_list.append(PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_))

    # 2. include (possible) activation
    if knots2_len > 1:
        nets_list.append(DistConvertor_(knots2_len, symmetric=zee2sym, smooth=True))

    # 3. Add (possible) affine blocks
    if acts is None:
        tag = "tanh" if zee2sym else "leaky_relu"
        acts = (*[tag] * len(hidden_sizes), None)

    conv_dict = dict(
        in_channels=1,
        out_channels=2,
        hidden_sizes=hidden_sizes,
        kernel_size=3,
        padding_mode="circular",
        conv_dim=len(lat_shape),
        acts=acts,
        bias=not zee2sym,
    )
    mask = EvenOddMask(shape=lat_shape)
    nets_list.append(
        AffineCoupling_([ConvAct(**conv_dict) for _ in range(n_layers)], mask=mask)
    )

    # 4. include (possible) activation
    if knots4_len > 1:
        nets_list.append(DistConvertor_(knots4_len, symmetric=zee2sym, smooth=True))

    return nets_list
