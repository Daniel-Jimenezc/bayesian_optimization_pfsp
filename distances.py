import warnings
from functools import partial

import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.spatial import _distance_wrap
from scipy.spatial.distance import _args_to_kwargs_xdist, _METRIC_ALIAS, _filter_deprecated_kwargs, _METRICS, \
    mahalanobis, wminkowski, minkowski, seuclidean, _validate_cdist_input, _select_weighted_metric, _C_WEIGHTED_METRICS, \
    _TEST_METRICS, squareform, _validate_pdist_input, jensenshannon, _convert_to_type

_convert_to_double = partial(_convert_to_type, out_type=np.double)
from numba import njit



@njit
def kendall_distance_original(x,y):
    distance = 0
    for i in range(len(x)):
        for j in range(i,len(x)):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if (a * b < 0):
                distance += 1
    return distance

@njit
def calculate_pdist_dm(metric,dm,m,X):
    k = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dm[k] = metric(X[i], X[j])
            k = k + 1
    return dm

@njit
def calculate_cdist_dm(metric,dm,mA,mB,XA,XB):
    for i in range(0, mA):
        for j in range(0, mB):
            dm[i, j] = metric(XA[i], XB[j])
    return dm


def custom_pdist(X, metric='euclidean', *args, **kwargs):
    X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
                           check_finite=False)
    kwargs = _args_to_kwargs_xdist(args, kwargs, metric, "pdist")

    X = np.asarray(X, order='c')

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s
    out = kwargs.pop("out", None)
    if out is None:
        dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    else:
        if out.shape != (m * (m - 1) // 2,):
            raise ValueError("output array has incorrect shape.")
        if not out.flags.c_contiguous:
            raise ValueError("Output array must be C-contiguous.")
        if out.dtype != np.double:
            raise ValueError("Output array must be double type.")
        dm = out

    # compute blocklist for deprecated kwargs
    if(metric in _METRICS['jensenshannon'].aka
       or metric == 'test_jensenshannon' or metric == jensenshannon):
        kwargs_blocklist = ["p", "w", "V", "VI"]

    elif(metric in _METRICS['minkowski'].aka
         or metric in _METRICS['wminkowski'].aka
         or metric in ['test_minkowski', 'test_wminkowski']
         or metric in [minkowski, wminkowski]):
        kwargs_blocklist = ["V", "VI"]

    elif(metric in _METRICS['seuclidean'].aka or
         metric == 'test_seuclidean' or metric == seuclidean):
        kwargs_blocklist = ["p", "w", "VI"]

    elif(metric in _METRICS['mahalanobis'].aka
         or metric == 'test_mahalanobis' or metric == mahalanobis):
        kwargs_blocklist = ["p", "w", "V"]

    else:
        kwargs_blocklist = ["p", "V", "VI"]

    _filter_deprecated_kwargs(kwargs, kwargs_blocklist)

    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_name = _METRIC_ALIAS.get(mstr, None)

        if metric_name is not None:
            X, typ, kwargs = _validate_pdist_input(X, m, n,
                                                   metric_name, **kwargs)

        dm = calculate_pdist_dm(metric,dm,m,X)

    elif isinstance(metric, str):
        mstr = metric.lower()

        mstr, kwargs = _select_weighted_metric(mstr, kwargs, out)

        metric_name = _METRIC_ALIAS.get(mstr, None)

        if metric_name is not None:
            X, typ, kwargs = _validate_pdist_input(X, m, n,
                                                   metric_name, **kwargs)

            if 'w' in kwargs:
                metric_name = _C_WEIGHTED_METRICS.get(metric_name, metric_name)

            # get pdist wrapper
            pdist_fn = getattr(_distance_wrap,
                               "pdist_%s_%s_wrap" % (metric_name, typ))
            pdist_fn(X, dm, **kwargs)
            return dm

        elif mstr in ['old_cosine', 'old_cos']:
            warnings.warn('"old_cosine" is deprecated and will be removed in '
                          'a future version. Use "cosine" instead.',
                          DeprecationWarning)
            X = _convert_to_double(X)
            norms = np.einsum('ij,ij->i', X, X, dtype=np.double)
            np.sqrt(norms, out=norms)
            nV = norms.reshape(m, 1)
            # The numerator u * v
            nm = np.dot(X, X.T)
            # The denom. ||u||*||v||
            de = np.dot(nV, nV.T)
            dm = 1.0 - (nm / de)
            dm[range(0, m), range(0, m)] = 0.0
            dm = squareform(dm)
        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                dm = custom_pdist(X, _TEST_METRICS[mstr], **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm


def custom_cdist(XA, XB, metric='euclidean', *args, **kwargs):
    kwargs = _args_to_kwargs_xdist(args, kwargs, metric, "cdist")

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    out = kwargs.pop("out", None)
    if out is None:
        dm = np.empty((mA, mB), dtype=np.double)
    else:
        if out.shape != (mA, mB):
            raise ValueError("Output array has incorrect shape.")
        if not out.flags.c_contiguous:
            raise ValueError("Output array must be C-contiguous.")
        if out.dtype != np.double:
            raise ValueError("Output array must be double type.")
        dm = out

    # compute blocklist for deprecated kwargs
    if(metric in _METRICS['minkowski'].aka or
       metric in _METRICS['wminkowski'].aka or
       metric in ['test_minkowski', 'test_wminkowski'] or
       metric in [minkowski, wminkowski]):
        kwargs_blocklist = ["V", "VI"]
    elif(metric in _METRICS['seuclidean'].aka or
         metric == 'test_seuclidean' or metric == seuclidean):
        kwargs_blocklist = ["p", "w", "VI"]
    elif(metric in _METRICS['mahalanobis'].aka or
         metric == 'test_mahalanobis' or metric == mahalanobis):
        kwargs_blocklist = ["p", "w", "V"]
    else:
        kwargs_blocklist = ["p", "V", "VI"]

    _filter_deprecated_kwargs(kwargs, kwargs_blocklist)

    if callable(metric):

        mstr = getattr(metric, '__name__', 'Unknown')
        metric_name = _METRIC_ALIAS.get(mstr, None)

        XA, XB, typ, kwargs = _validate_cdist_input(XA, XB, mA, mB, n,
                                                    metric_name, **kwargs)

        dm = calculate_cdist_dm(metric,dm,mA,mB,XA,XB)

    elif isinstance(metric, str):
        mstr = metric.lower()

        mstr, kwargs = _select_weighted_metric(mstr, kwargs, out)

        metric_name = _METRIC_ALIAS.get(mstr, None)
        if metric_name is not None:
            XA, XB, typ, kwargs = _validate_cdist_input(XA, XB, mA, mB, n,
                                                        metric_name, **kwargs)

            if 'w' in kwargs:
                metric_name = _C_WEIGHTED_METRICS.get(metric_name, metric_name)

            # get cdist wrapper
            cdist_fn = getattr(_distance_wrap,
                               "cdist_%s_%s_wrap" % (metric_name, typ))
            cdist_fn(XA, XB, dm, **kwargs)
            return dm

        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                dm = custom_cdist(XA, XB, _TEST_METRICS[mstr], **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm