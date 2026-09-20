"""Publication style helper + palette."""

from electropycal.viz import OKABE_ITO, categorical, set_pub_style


def test_categorical_fixed_order_no_cycle_within_palette():
    assert categorical(3) == OKABE_ITO[:3]
    assert len(categorical(8)) == 8
    assert len(set(categorical(8))) == 8              # 8 distinct within the palette


def test_set_pub_style_applies_rcparams():
    import matplotlib as mpl
    set_pub_style()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["legend.frameon"] is False
    # categorical prop_cycle is the Okabe-Ito order
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][:2] == OKABE_ITO[:2]
