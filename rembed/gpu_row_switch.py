import theano
from theano import gof
from theano import tensor as T
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (CudaNdarrayType, GpuFromHost,
                                           HostFromGpu,
                                           as_cuda_ndarray_variable)
from theano.tensor.opt import register_specialize_device


class GpuRowSwitch(GpuOp):
    """
    Row-wise switch between rank-2 matrices on the GPU.

    DOES NOT support broadcasting arguments (e.g. T.switch(mask, A, 0)).

    >>> A
    [[ 0.01902644  0.70658928  0.10509603]
     [ 0.2654964   0.08410256  0.96556276]
     [ 0.06885902  0.49623388  0.18812495]
     [ 0.56566966  0.52721274  0.48890418]]
    >>> B
    [[ 0.44089654  0.46353787  0.59428871]
     [ 0.88936949  0.74785614  0.80535758]
     [ 0.88973558  0.21844074  0.12561291]
     [ 0.01211281  0.86583334  0.9793455 ]]
    >>> mask
    [1 0 0 1]
    >>> GpuRowSwitch()(mask, A, B).eval()
    [[ 0.01902644  0.70658928  0.10509603]
     [ 0.88936949  0.74785614  0.80535758]
     [ 0.88973558  0.21844074  0.12561291]
     [ 0.56566966  0.52721274  0.48890418]]
    """

    nin = 3
    nout = 1

    def make_node(self, cond, ift, iff):
        if any(ift.broadcastable) or any(iff.broadcastable):
            raise ValueError("GPURowSwitch cannot operate on broadcastable "
                             "output arguments (ift %s, iff %s)."
                             % (ift.broadcastable, iff.broadcastable))
        out_type = ift.dtype

        cond = as_cuda_ndarray_variable(
                T.cast(cond.flatten(), "float32"))
        ift = as_cuda_ndarray_variable(ift)
        iff = as_cuda_ndarray_variable(iff)

        assert ift.type.dtype == iff.type.dtype
        assert cond.ndim == 1, cond.ndim
        assert ift.ndim == iff.ndim

        return gof.Apply(self, [cond, ift, iff],
                         [CudaNdarrayType(broadcastable=ift.broadcastable,
                                          dtype=out_type)()])

    def perform(self, node, inp, out):
        raise NotImplementedError("GPUSwitch is GPU only")

    def c_support_code(self):
        """Defines the abstract row-switching kernel used in this op."""

        return """
__global__ void k_row_switch(int ndim,
                             int shape1, int shape2, int shape3,
                             int stride1, int stride2, int stride3,
                             const float* cond, const float* ift,
                             const float* iff, float* out) {
  // batch index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 0 || idx >= shape1) {
    return;
  }

  const float *src = ((int) cond[idx]) ? ift : iff;
  int offset = idx * stride1;

  int lastDim = ndim == 2 ? shape2 : shape3;
  int lastStride = ndim == 2 ? stride2 : stride3;

  if (ndim == 3) {
      // index within the example
      int axis1_idx = blockIdx.y * blockDim.y + threadIdx.y;
      offset += axis1_idx * stride2;
  }

  for (int j = 0; j < lastDim; j++) {
    out[offset + j * lastStride] = src[offset + j * lastStride];
  }
  return;
}
""" % locals()

    def c_code(self, node, name, inp, out, sub):
        """Generates code to instantiate this op for these particular inputs."""

        cond, ift, iff = inp
        out, = out
        fail = sub["fail"]

        return """
int err, N, N2, ndim;
cudaError_t sts;
int threads_per_block1, n_blocks1;
int threads_per_block2 = 1, n_blocks2 = 1;
const int *dims, *strides;

N = CudaNdarray_SIZE(%(cond)s);
ndim = CudaNdarray_NDIM(%(ift)s);
switch (ndim) {
    case 3:
        N2 = CudaNdarray_HOST_DIMS(%(ift)s)[1];
        threads_per_block2 = std::min(N2, NUM_VECTOR_OP_THREADS_PER_BLOCK);
        n_blocks2 = std::min(NUM_VECTOR_OP_BLOCKS,
                             (N2 + NUM_VECTOR_OP_THREADS_PER_BLOCK - 1) / NUM_VECTOR_OP_THREADS_PER_BLOCK);
        // NB: no break!
    case 2:
        threads_per_block1 = std::min(N, NUM_VECTOR_OP_THREADS_PER_BLOCK);
        n_blocks1 = std::min(NUM_VECTOR_OP_BLOCKS,
                             (N + NUM_VECTOR_OP_THREADS_PER_BLOCK - 1) / NUM_VECTOR_OP_THREADS_PER_BLOCK);
        break;
    default:
        return 1;
}

dim3 n_blocks(n_blocks1, n_blocks2);
dim3 threads_per_block(threads_per_block1, threads_per_block2);

// Allocate the output array
Py_XDECREF(%(out)s);
%(out)s = (CudaNdarray *) CudaNdarray_NewDims(ndim, CudaNdarray_HOST_DIMS(%(ift)s));
if (!%(out)s) {
    %(fail)s;
}


dims = CudaNdarray_DIMS(%(ift)s);
strides = CudaNdarray_STRIDES(%(ift)s);

// Instantiate the kernel.
//
// TODO: Assumes stride of ift, iff are the same
k_row_switch<<<n_blocks, threads_per_block>>>(
    ndim,
    dims[0], dims[1], dims[2],
    strides[0], strides[1], strides[2],
    CudaNdarray_DEV_DATA(%(cond)s),
    CudaNdarray_DEV_DATA(%(ift)s),
    CudaNdarray_DEV_DATA(%(iff)s),
    CudaNdarray_DEV_DATA(%(out)s));

// Force async kernel instances to sync at this thread barrier
CNDA_THREAD_SYNC;

sts = cudaGetLastError();
if (cudaSuccess != sts) {
    PyErr_Format(PyExc_RuntimeError, "Cuda error: k_row_switch: %%s.",
                 cudaGetErrorString(sts));
    %(fail)s;
}
""" % locals()


##################################


@register_specialize_device("row_switch")
@gof.local_optimizer([theano.tensor.Elemwise, theano.scalar.Switch])
def local_gpua_row_switch(node):
    """
    Detects eligible Switch instances and replaces them with a GPU
    row switch.
    """

    if (node.op.__class__ == theano.tensor.Elemwise
        and node.op.scalar_op.__class__ != theano.scalar.Switch):
        return False

    cond, ift, iff = node.inputs
    out, = node.outputs

    # Only applies to Switch instances where a vector mask broadcasts over
    # matrices.
    bcast = cond.broadcastable
    if not (len(bcast) > 1 and not bcast[0] and all(bcast[1:])
            and ift.ndim in [2, 3]):
        return False

    if not (ift.dtype == iff.dtype == "float32"):
        return False

    if cond.owner and isinstance(cond.owner.op, HostFromGpu):
        gpu_cond, = cond.owner.inputs
    else:
        gpu_cond = as_cuda_ndarray_variable(
                T.cast(cond.flatten(), "float32"))

    if ift.owner and isinstance(ift.owner.op, HostFromGpu):
        gpu_ift, = ift.owner.inputs
    else:
        gpu_ift = as_cuda_ndarray_variable(ift)

    if iff.owner and isinstance(iff.owner.op, HostFromGpu):
        gpu_iff, = iff.owner.inputs
    else:
        gpu_iff = as_cuda_ndarray_variable(iff)

    gpu_op = GpuRowSwitch()
    return [HostFromGpu()(gpu_op(cond, gpu_ift, gpu_iff))]


######################

import numpy as np

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

# jagupard10$ THEANO_FLAGS=device=gpu7,floatX=float32 python if_positive.py
# Couldn't import dot_parser, loading of dot files will not be possible.
# Using gpu device 7: GeForce GTX TITAN X (CNMeM is disabled)
# Elemwise{switch,no_inplace} [@A] ''
#  |DimShuffle{0,x} [@B] ''
#  | |mask [@C]
#  |HostFromGpu [@D] ''
#  | |A [@E]
#  |HostFromGpu [@F] ''
#    |B [@G]
#
#
#
#
# HostFromGpu [@A] ''
#  |<__main__.GpuRowSwitch object at 0x7f8335f9b810> [@B] ''
#    |GpuFromHost [@C] ''
#    | |Elemwise{Cast{float32}} [@D] ''
#    |   |Flatten{1} [@E] ''
#    |     |InplaceDimShuffle{0,x} [@F] ''
#    |       |mask [@G]
#    |A [@H]
#    |B [@I]
# [[ 0.9905898   0.7576282   0.02603905 ...,  0.52712584  0.91501969
#    0.99620157]
#  [ 0.654755    0.14092255  0.53908485 ...,  0.817927    0.08480006
#    0.68436199]
#  [ 0.42901123  0.05686407  0.90083718 ...,  0.67377824  0.71434939
#    0.93992913]
#  [ 0.59950465  0.5745737   0.08946238 ...,  0.73683125  0.12341607
#    0.2341859 ]]
# [[ 0.47230095  0.94467312  0.53877729 ...,  0.27510449  0.65054226
#    0.62130648]
#  [ 0.51275951  0.37535876  0.58075702 ...,  0.74825442  0.86357826
#    0.8180939 ]
#  [ 0.18833372  0.94145817  0.24612874 ...,  0.76728404  0.03735197
#    0.75683349]
#  [ 0.70569879  0.14489071  0.16729677 ...,  0.27402359  0.48268986
#    0.79868871]]
# [1 0 0 0]
#
#
#
#
# [[ 0.9905898   0.7576282   0.02603905 ...,  0.52712584  0.91501969
#    0.99620157]
#  [ 0.51275951  0.37535876  0.58075702 ...,  0.74825442  0.86357826
#    0.8180939 ]
#  [ 0.18833372  0.94145817  0.24612874 ...,  0.76728404  0.03735197
#    0.75683349]
#  [ 0.70569879  0.14489071  0.16729677 ...,  0.27402359  0.48268986
#    0.79868871]]
