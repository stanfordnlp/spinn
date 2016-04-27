import numpy as np
import theano
import theano.tensor as T

# Import cuda util in order to register the optimization
from spinn.util import cuda


def _test_gpu_rowwise_switch_inner(f, A, B, mask, expected):
    ret = f(A, B, mask)
    print A
    print B
    print mask
    print ret
    print expected
    np.testing.assert_array_almost_equal(ret, expected)


def test_gpu_rowwise_switch():
    assert theano.config.device.startswith("gpu"), "Need to test on GPU!"

    data = [
        # 4 x 2
        (np.array([[ 0.22323515,  0.36703175],
                   [ 0.82260513,  0.3461504 ],
                   [ 0.82362652,  0.81626087],
                   [ 0.95270008,  0.2226797 ]]),
         np.array([[ 0.36341551,  0.20102882],
                   [ 0.24144639,  0.45237923],
                   [ 0.39951822,  0.7348066 ],
                   [ 0.16649647,  0.60306537]]),
         np.array([1, 0, 1, 1]),
         np.array([[ 0.22323515,  0.36703175],
                   [ 0.24144639,  0.45237923],
                   [ 0.82362652,  0.81626087],
                   [ 0.95270008,  0.2226797 ]])),

        # 2 x 3 x 4
        (np.array([[[ 0.48769062,  0.82649632,  0.2047115 ,  0.41437615],
                    [ 0.25290664,  0.87164914,  0.80968588,  0.49295084],
                    [ 0.71438099,  0.97913502,  0.37598001,  0.76958707]],

                   [[ 0.37605973,  0.538358  ,  0.74304674,  0.84346291],
                    [ 0.95310617,  0.61540292,  0.49881143,  0.1028554 ],
                    [ 0.83481996,  0.90969569,  0.40410424,  0.34419989]]]),
         np.array([[[ 0.7289117 ,  0.97323253,  0.19070121,  0.64164653],
                    [ 0.26816493,  0.76093069,  0.95284825,  0.77350426],
                    [ 0.55415519,  0.39431256,  0.86588665,  0.50031027]],

                   [[ 0.1980869 ,  0.7753601 ,  0.26810868,  0.3628802 ],
                    [ 0.2488143 ,  0.21278388,  0.09724567,  0.58457886],
                    [ 0.12295105,  0.75321368,  0.37258797,  0.27756972]]]),
         np.array([1, 0]),
         np.array([[[ 0.48769062,  0.82649632,  0.2047115 ,  0.41437615],
                    [ 0.25290664,  0.87164914,  0.80968588,  0.49295084],
                    [ 0.71438099,  0.97913502,  0.37598001,  0.76958707]],

                   [[ 0.1980869 ,  0.7753601 ,  0.26810868,  0.3628802 ],
                    [ 0.2488143 ,  0.21278388,  0.09724567,  0.58457886],
                    [ 0.12295105,  0.75321368,  0.37258797,  0.27756972]]]))

    ]

    A2, B2 = T.matrices("AB")
    A3, B3 = T.tensor3("A"), T.tensor3("B")
    mask = T.ivector("mask")

    switch2 = T.switch(mask.dimshuffle(0, "x"), A2, B2)
    switch3 = T.switch(mask.dimshuffle(0, "x", "x"), A3, B3)

    f2 = theano.function([A2, B2, mask], switch2)
    f3 = theano.function([A3, B3, mask], switch3)

    print "Graph of 2dim switch:"
    theano.printing.debugprint(f2.maker.fgraph.outputs[0])
    print "Graph of 3dim switch:"
    theano.printing.debugprint(f3.maker.fgraph.outputs[0])

    for instance in data:
        # Retrieve appropriate function
        func = f2 if instance[0].ndim == 2 else f3

        # Cast to float-friendly types
        instance = [x.astype(np.float32) if x.dtype.kind == 'f'
                    else x.astype(np.int32) for x in instance]

        yield tuple([_test_gpu_rowwise_switch_inner, func] + instance)


def _test_masked_careduce_inner(f, X, Y, mask, expected):
    print X
    print Y
    print mask

    ret = f(X, Y, mask)
    print ret
    print expected
    np.testing.assert_array_almost_equal(ret, expected)


def test_masked_careduce():
    assert theano.config.device.startswith("gpu"), "Need to test on GPU!"

    data = [
        (np.array([[[ 3.,  1.],
                    [ 4.,  8.]],

                   [[ 9.,  4.],
                    [ 3.,  6.]],

                   [[ 5.,  2.],
                    [ 6.,  2.]]]),

         np.array([[[ 10.,   7.],
                    [   7,   5.]],

                   [[  2.,   1.],
                    [  5.,   9.]],

                   [[  0.,   6.],
                    [  3.,   4.]]]),

         np.array([0, 1, 0]),

         np.array([[ 19.,  17.],
                   [ 13.,  15.]]))
    ]

    X, Y = T.tensor3("X"), T.tensor3("Y")
    mask = T.fvector("mask")
    mask_ = mask.dimshuffle(0, "x", "x")

    switch = T.switch(mask_, X, Y)
    out = switch.sum(axis=0)

    f = theano.function([X, Y, mask], out)

    print "Graph of switch+sum:"
    theano.printing.debugprint(f.maker.fgraph.outputs[0])

    for instance in data:
        # Cast to float-friendly types
        instance = [x.astype(np.float32) for x in instance]
        yield tuple([_test_masked_careduce_inner, f] + instance)

