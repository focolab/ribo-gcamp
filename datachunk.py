import copy
import pdb
import itertools
import json
import os
import base64

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import numpy as np

def chunk_ix(shape=None, dims=None, req=None):
    """indices for a chunk request (expanding a chunk request)

    'shape' and 'dims' describe the shape and dims names of an array A of
    arbitrary dimensions. 'req' is a (compact, dictionary) request for a chunk
    of A (np.ndarray).

    arguments
    ------
    shape (list): array shape
    dims (list or str): array dims ('XYZTC' or ['X','Y','Z','T','C'])
    req (dict): chunk request (e.g. {'X':1, 'T':[0,2], 'Y':(0,5)})

    req format options:
        X=1             interpreted as [1]
        X=(0,3)         interpreted as range(0,3)
        X=[0,3]         interpreted as itself (a list)
        X=slice(0,3)    python slice (e.g. slice(None,None,10) for every 10th)
        X=None          (default) all values

    returns
    ------
    ix (list):  a list of lists of index values for each dimension. Using the 
                output, numpy.ix_(ix) is a valid slice of A. Similarly, 
                itertools.product(*ix) makes a list of tuples for chunk
                elements in A.

    TODO: how to handle missing dims in the request? Return all/zero/error?
            A: if size is 1, use it, otherwise
    """

    #== check if requested dims DNE in the data
    for k in req.keys():
        if k not in dims:
            raise Exception('requested axis (%s) not in dims (%s)' % (k, str(dims)))

    #== a partial request, missing some dims, is made explicit
    reqloc = {k:v for k,v in req.items()}
    for a in dims:
        if a not in req.keys():
            reqloc[a] = None
            #print('WARNING: request set to None for %s' % a)

    ix = []
    for size, dim in zip(shape, dims):
        r = reqloc[dim]
        if isinstance(r, int):
            ix.append([r])
        elif isinstance(r, tuple):
            ix.append(list(range(*r)))
        elif isinstance(r, list):
            ix.append(r)
        elif isinstance(r, slice):
            ix.append(range(size)[r])
        elif r is None:
            ix.append(list(range(size)))
        else:
            raise Exception('chunk index: (%s) not recognized' %
                            (str(reqloc[dim])))

    return ix

def make_figax(Nx=0, Ny=0, Nz=0, stackmode='h'):
    """make multipanel figure with scaled axes

    TODO: this is orphaned needs a new home
    """

    #== figure setup and scaling
    h2w = Ny/Nx
    if stackmode == 'h':
        aspect = h2w/Nz
        fig = plt.figure(figsize=plt.figaspect(aspect))
        ax = [plt.subplot(1, Nz, i+1) for i in range(Nz)]
        for i, axi in enumerate(ax):
            if i>0:
                axi.set_yticklabels([])
    elif stackmode == 'v':
        aspect = h2w*Nz
        fig = plt.figure(figsize=plt.figaspect(aspect))
        ax = [plt.subplot(Nz, 1, i+1) for i in range(Nz)]
        for i, axi in enumerate(ax):
            if i<Nz-1:
                axi.set_xticklabels([])

    return fig, ax


class DataChunk(object):
    """multidimensional data and metadata

    TODO: coords attribute (echoing xarray)
    TODO: global channel info
    TODO: chunk type attribute (montage, etc..)
    TODO: axes values and units (maybe x, y, z for X Y Z ())
    TODO: track an offset, possibly a stride too
    TODO: global vs local properties? (i.e. global intensity limits vs local)

    json en/de-coding
    https://stackoverflow.com/a/6485943/6474403

    DONE :)
    TODO: axes should be renamed dims
    TODO: to dict/json/file from dict/json/file
    TODO: remove shape from the constructor, infer it from data
    """

    def __init__(self, data, **kwargs):

        self.data = data
        self.dims = kwargs.get('dims', None)

        self.meta = kwargs.get('meta', {})

        if kwargs.get('axes', None) is not None:
            raise Exception('axes no longer accepted, used dims instead')

        #== derived
        self.dim_len = dict(zip(self.dims,self.data.shape))

        self._channel_ranges = None


    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def channel_ranges(self, dim='C'):

        if self._channel_ranges is None:
            ranges = []
            for ic in range(self.dim_len[dim]):
                data = self.subchunk(req={dim:ic}).data.ravel()
                ranges.append([np.min(data), np.max(data)])
            self._channel_ranges = ranges
        return self._channel_ranges


    def about(self, verbose=False):
        """report about this DataChunk"""
        print('dtype    :', self.dtype)
        print('dims     :', self.dims)
        print('shape    :', self.shape)
        print('meta     :')
        for k, v in self.meta.items():
            if len(str(v))>120:
                beg = str(v)[:60]
                end = str(v)[-60:]
                print('  %10s : %s' % (str(k), beg))
                print('  %10s   %s' % ('', '...'))
                print('  %10s   %s' % ('', end))

            else:
                print('  %10s : %s' % (str(k), str(v)))
        if verbose:
            print('data     :')
            print(self.data)

    def squeeze(self, inplace=False):
        """drop dimensions with shape==1

        TODO: put squeezed out dimensions into meta
        """
        if inplace:
            raise Exception('inplace not yet implemented')

        data = np.squeeze(self.data)
        keep = np.where(np.asarray(self.shape) > 1)[0]
        dims = [self.dims[k] for k in keep]
        md = self.meta

        return DataChunk(data=data, dims=dims, meta=md)

    def reorder_dims(self, dims_new, inplace=False, verbose=False):
        """reorder ALL dims (have to specify them all)

        TODO: make shortcuts (leading, trailing) that call this
        TODO: np.transpose might be cleaner
        """
        if inplace:
            raise Exception('inplace=True not yet implemented')

        for ax in dims_new:
            if ax not in self.dims:
                raise Exception('bogus request, (%s) not found in existing dims (%s)'
                                % (str(ax), str(self.dims)))

        ndx = list(range(len(self.dims)))       #== source dims to grab
        newloc = dict(zip(dims_new, ndx))       #== new locations, by name
        dst = [newloc[a] for a in self.dims]    #== new locations, by ndx

        if verbose:
            print('ndx:', ndx, self.dims)
            print('dst:', dst, dims_new)

        data = np.moveaxis(self.data, ndx, dst)
        md = self.meta.copy()

        return DataChunk(data=data, dims=dims_new, meta=md)

    def subchunk(self, req=None, inplace=False, squeeze=False, verbose=False):
        """return a subchunk of a data (also a DataChunk)

        chunk indexing format options:
            X=1         interpreted as index 1
            X=(0,3)     interpreted as a range
            X=[0,3]     interpreted as a list
            X=None      all values
        """

        if inplace:
            raise Exception('inplace=True not yet implemented')

        #== building the chunk request
        req_ix = chunk_ix(shape=self.shape, dims=self.dims, req=req)
        ix = np.ix_(*req_ix)

        chunk = self.data[ix]
        dims = self.dims
        md = self.meta.copy()
        #md['generation'] = 2

        if verbose:
            print('-----------')
            print('dims   :', self.dims)
            print('shape   :', self.shape)
            print('request:', req)
            print('ix     :', ix)
            print('chunk  :', chunk)
            #print(np.squeeze(chunk))
            print('-----------')

        return DataChunk(data=chunk, dims=dims, meta=md)

    def montage(self, I=None, J=None, II='', JJ='', T='', C=''):
        """
        stack (images/frames/panels) horizontally and/or vertically

        The canonical case is to stack together multiple XY frames
        (I,J indexed), to show multiple Z-planes or time points

        This method allows horizontal and vertical stacking, where II and JJ
        denote the outer stack dimensions. I and J denote inner dimensions of
        a single frame (possibly with a 3rd color dimesion, indicated with C).
        I/II and J/JJ are row and column directions, respectively

        I,J,II,JJ,C,T have the following roles here:
            I: frame/inner vertical axis (row)
            J: frame/inner horizontal axis (column)
            II: stack/outer vertical axis (row)
            JJ: stack/outer horizontal axis (column)
            C: (optional) color axis for multichannel datasets
            T: (optional) time/animate axis (a movie has one (I,II,J,JJ) per T)

        NOTES:  This should not swamp memory, provided the parent DataChunk
                comfortably fits in memory. To dump or stream big montages,
                consider using multiple DataChunks (probably split on the time
                dimension), rather than one colossal chunk.

                This is mainly for visualization/output, not downsteam
                analysis. The stacked dimensions are not set up for easy
                manipulation. Coords would have to be generalized for the
                stacked dimensions, reminiscent of a pandas multi-index
                dataframe. Currently, the metadata 'stacked_dims' is just sort
                of dumped into meta without a real future plan.

                The standalone function montage_labels(chunk) takes a montage
                (DataChunk) and returns indices and label strings for labeling
                a montage image (tested assuming matplotlib.imshow())

        RETURNS
        ------
        a DataChunk with data in TIJC dimension order. This allows direct
        plotting via Matplotlib imshow, for example, which requires (M,N) or
        (M,N,3) data for monochrome/rgb cases.

        TODO: this is a specific case of a more general data reshaping process
        TODO: carry stuff forward (global color channel limits, coords, meta)
        TODO: COORDS GETS BORKED (Y,Z -> 'Y:Z' ? )
        TODO: make an iterator that streams chunks..
        """

        if 'montage' in self.meta.keys():
            raise Exception('cannot montage twice, you weirdo.. start over')

        # verify the request is legit
        for ax in [I, J, II, JJ, T, C]:
            if ax is not '' and ax not in self.dims:
                raise Exception('bogus montage request, (%s) not found in existing dims (%s)'
                                % (str(ax), str(self.dims)))

        # reorder dimensions and make a new chunk
        req_reorder = T+II+JJ+I+J+C
        chunk = self.reorder_dims(req_reorder) #, verbose=True)

        # convenience, data sizes
        Ni = chunk.dim_len.get(I, 1)
        Nj = chunk.dim_len.get(J, 1)
        Nii = chunk.dim_len.get(II, 1)
        Njj = chunk.dim_len.get(JJ, 1)
        Nt = chunk.dim_len.get(T, 1)


        # assemble the montage data, could be accelerated
        data = []
        for it in range(Nt):
            frames = []
            req = {T:it} if Nt>1 else {}
            paneldata = []
            for row in range(Nii):
                if Nii > 1:
                    req[II] = row
                rowdata = []
                for col in range(Njj):
                    req[JJ] = col
                    rowdata.append(chunk.subchunk(req=req).squeeze().data)
                paneldata.append(np.hstack(rowdata))
            data.append(np.vstack(paneldata))
        data = np.asarray(data)


        # update metadata with montage info
        meta = {k:v for k,v in self.meta.items()}

        md = dict(panel_width=Nj,
                  panel_height=Ni,
                  dim_JJ=JJ,
                  dim_II=II,
                  num_JJ=Njj,
                  num_II=Nii,
                  dim_J=J,
                  dim_I=I,
                  num_J=Nj,
                  num_I=Ni,
                  is2D=False)

        if Nii>1 and Njj>1:
            md['is2D'] = True
            md['stackmode'] = '2D'
        elif Nii>1:
            md['stackmode'] = 'v'
        elif Njj>1:
            md['stackmode'] = 'h'
        else:
            raise Exception('Nii<1 and Njj<1.. wtf??')

        meta['montage'] = md


        # build the output DataChunk, squeeze out extra dimensions
        dims = [T, I, J, C]
        return DataChunk(data=data, dims=dims, meta=meta).squeeze()



    def to_jdict(self, data_encoding=None):
        """json compatible dictionary

        data_encoding (str): 'raw' or 'b64str'
        """

        jdict = dict(_about='a DataChunk in json/dictionary form',
                     dims=self.dims,
                     shape=self.shape,
                     dtype=str(self.dtype),
                     meta=self.meta)

        if data_encoding in [None, 'raw']:
            #== serialize the data into a 1D list
            dsrl = self.data.ravel().tolist()
            jdict['data_1D'] = dsrl
        else:
            #== encode data as a base64 string
            data_b64 = base64.b64encode(self.data.ravel())
            data_b64str = data_b64.decode('utf-8')
            jdict['data_b64str'] = data_b64str

        return jdict

    @classmethod
    def from_jdict(cls, d):
        """alternate constructor, using json compatible dictionary"""

        dims = d.get('dims', None)
        shape = d.get('shape', None)
        meta = d.get('meta', None)
        dtype = d.get('dtype', None)

        #== decode and reshape the serialized data
        if 'data_b64str' in d.keys():
            data_b64str = d['data_b64str']
            dd = base64.b64decode(data_b64str)
            data = np.frombuffer(dd, dtype=dtype).reshape(shape)
        elif 'data_1D' in d.keys():
            dsrl = d['data_1D']
            data = np.asarray(dsrl, dtype=dtype).reshape(shape)
        else:
            err = 'no data in the jdict, need either data_1D or data_b64str'
            raise Exception(err)

        kwa = dict(dims=dims,
                   meta=meta,
                   data=data)

        return cls(**kwa)

    def to_json(self, out='chunk.json', make_path=True,
                data_encoding=None):
        """dump DataChunk to a json file"""

        if make_path:
            os.makedirs(os.path.dirname(out), exist_ok=True)

        jdict = self.to_jdict(data_encoding=data_encoding)
        with open(out, 'w') as f:
            json.dump(jdict, f, indent=2)
            f.write('\n')

    @classmethod
    def from_json(cls, j):
        """alternate constructor, from json file"""

        with open(j) as jfopen:
            jdict = json.load(jfopen)

        return cls.from_jdict(jdict)


class ChunkMontage(object):
    def __init__(self):
        raise Exception('ChunkMontage DNE. Use DataChunk.montage()')
        return



def montage_labels(chunk=None):
    """generate panel labels for a montage

    NOTE: this REQUIRES coords..

    given a chunk with montage metadata, returns
        xypos: a list of upper left pixel coordinates for the panels
        vals: values of the montage coordinates (not really used)
        labels: string labels (e.g. 'Z=2 T=4') for each panel
    """

    coords = chunk.meta['coords']

    md = chunk.meta['montage']

    #== label positions
    xpos = md['panel_width']* np.arange(md.get('num_JJ', 1))
    ypos = md['panel_height']*np.arange(md.get('num_II', 1))
    xypos = itertools.product(xpos, ypos)

    #==
    if md['is2D']:
        dh = md.get('dim_JJ', '')
        dv = md.get('dim_II', '')
        xval = coords[dh]
        yval = coords[dv]
        xyval = itertools.product(xval, yval)
        vals = xyval
        labels = ['%s=%s \n%s=%s' % (dh, x, dv, y)  for x,y in xyval]

    else:
        if md['stackmode'] == 'h':
            dh = md.get('dim_JJ', '')
            xval = coords[dh]
            vals = xval
            labels = ['%s=%s' % (dh, x)  for x in xval]
        elif md['stackmode'] == 'v':
            dv = md.get('dim_II', '')
            yval = coords[dv]
            vals = yval
            labels = ['%s=%s' % (dv, y)  for y in yval]

    return xypos, vals, labels



#==============================================================================
#                            TEST DEMARCATION
#               everything below is tests, above is code
#==============================================================================

def test_datachunk_montage(doPlots=True, verbose=False):
    """testing DataChunk.montage()

    monochrome:
    - multiple XYZ montages, one per T
    - single TXYZ montage

    rgb:
    - multiple XYZC montages, one per T
    - single TXYZ montage
    - double montage (TXYZC -> XYC, with T and Z stacked on X and Y dims)

    """

    fldr = os.path.join(os.path.dirname(__file__), 'scratch')
    os.makedirs(fldr, exist_ok=True)

    #==========================================================
    #== monochrome image hyperstack
    shape = [4, 5, 1, 20, 6]
    dims = 'TZCYX'
    data = np.arange(np.product(shape)).reshape(shape)
    dc = DataChunk(data=data, dims=dims)
    ranges = dc.channel_ranges


    # make multiple XYZ montages, one for each T
    for it in range(4):
        cm = dc.subchunk(req={'T':it}).squeeze().montage(I='Y', J='X', JJ='Z')

        assert cm.shape == (20, 30), "shape should be (20,30)"
        assert cm.dims == ['Y', 'X'], "dims should be ['Y', 'X']"

        if doPlots:
            svg = os.path.join(fldr, 'plot-testmontageA-%4.4i.svg' % (it))
            plt.imshow(cm.data, vmin=ranges[0][0], vmax=ranges[0][1])
            plt.xlabel('X coord, Z-stacked, should be (6px)*5')
            plt.ylabel('Y coord, should be 20px')
            plt.savefig(svg, dpi=400)
            plt.clf()


    # make a single TXYZ montage
    cm = dc.squeeze().montage(T='T', I='X', J='Y', II='Z')

    assert cm.shape == (4, 30, 20), "shape should be (4, 30, 20)"
    assert cm.dims == ['T', 'X', 'Y'], "dims should be ['T', 'X', 'Y']"

    for it in range(4):
        data = cm.data[it]

        assert data.shape == (30, 20), "data.shape should be (30,20)"

        if doPlots:
            svg = os.path.join(fldr, 'plot-testmontageB-%4.4i.svg' % (it))
            plt.imshow(data, vmin=ranges[0][0], vmax=ranges[0][1])
            plt.xlabel('Y coord, should be 20px')
            plt.ylabel('X coord, Z-stacked, should be (6px)*5')
            plt.savefig(svg, dpi=400)
            plt.clf()


    #==========================================================
    #== RGB

    # make data (not war)
    shape = [4, 5, 3, 20, 6]
    dims = 'TZCYX'

    data = np.arange(np.product(shape)).reshape(shape)
    data = np.floor(data/np.max(data.ravel())*255).astype(np.uint8)

    tt = np.asarray([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    aa = np.asarray([[1,1], [1,1]])

    for it in range(4):
        for iz in range(5):
            data[it,iz,0, 1:5, 1:5] = tt*100
            data[it,iz,1, 11:15, 1:5] = tt*180
            data[it,iz,2, 0:2, 0:2] = aa*150
            data[it,iz,2, -2:, -2:] = aa*150
    dc = DataChunk(data=data, dims=dims)


    # make multiple ZYXC montages, one for each T
    for it in range(4):
        cm = dc.subchunk(req={'T':it}).squeeze().montage(I='Y', J='X', JJ='Z', C='C')

        assert cm.shape == (20, 30, 3), "shape should be (20,30,3)"
        assert cm.dims == ['Y', 'X', 'C'], "dims should be ['Y', 'X', 'C']"

        if doPlots:
            svg = os.path.join(fldr, 'plot-testmontage-rgb-A-%4.4i.svg' % (it))
            plt.imshow(cm.data)
            plt.xlabel('X coord, Z-stacked, should be (6px)*5')
            plt.ylabel('Y coord, should be 20px')
            plt.savefig(svg, dpi=400)
            plt.clf()


    # one big montage
    cm = dc.squeeze().montage(T='T', I='Y', J='X', JJ='Z', C='C')

    assert cm.shape == (4, 20, 30, 3), "shape should be (4, 20, 30, 3)"
    assert cm.dims == ['T', 'Y', 'X', 'C'], "dims should be ['T', 'Y', 'X', 'C']"

    for it in range(4):
        data = cm.data[it]

        if doPlots:
            svg = os.path.join(fldr, 'plot-testmontage-rgb-B-%4.4i.svg' % (it))
            plt.imshow(data)
            plt.xlabel('X coord, Z-stacked, should be (6px)*5')
            plt.ylabel('Y coord, should be 20px')
            plt.savefig(svg, dpi=400)
            plt.clf()



    # montaged montage number 1
    cmm = dc.squeeze().montage(I='Y', J='X', JJ='Z', II='T', C='C')

    assert cmm.shape == (80, 30, 3), "shape should be (80, 30, 3)"
    assert cmm.dims == ['Y', 'X', 'C'], "dims should be ['Y', 'X', 'C']"

    if doPlots:
        svg = os.path.join(fldr, 'plot-testmontage-uberA.svg')
        plt.imshow(cmm.data)
        plt.xlabel('X coord, Z-stacked, should be (6px)*5')
        plt.ylabel('Y coord, T-stacked, should be (20px)*4')
        plt.savefig(svg, dpi=400)
        plt.clf()



    # montaged montage number 2
    cmm = dc.squeeze().montage(I='X', J='Y', JJ='T', II='Z', C='C')

    assert cmm.shape == (30, 80, 3), "shape should be (24, 80, 3)"
    assert cmm.dims == ['X', 'Y', 'C'], "dims should be ['X', 'Y', 'C']"

    if doPlots:
        svg = os.path.join(fldr, 'plot-testmontage-uberB.svg')
        plt.imshow(cmm.data)
        plt.xlabel('Y coord, T-stacked, should be (20px)*4')
        plt.ylabel('X coord, Z-stacked, should be (6px) *5')
        plt.savefig(svg, dpi=400)
        plt.clf()

    #print(cmm.meta)
    return



def test_datachunk_reorder_dims(verbose=False):
    """testing reordering the dims"""
    shape = [2, 4, 3]
    opts = dict(dims='XYZ')
    data = np.arange(np.product(shape)).reshape(shape)
    dc_XYZ = DataChunk(data=data, **opts)

    dc_ZYX = dc_XYZ.reorder_dims('ZYX')
    assert dc_ZYX.dims == 'ZYX', "should be ZYX"
    assert dc_ZYX.shape == (3, 4, 2), "should be (3, 4, 2)"

    #== I shall taunt (reorder) you a second time!!
    dc_YZX = dc_ZYX.reorder_dims('YZX')
    assert dc_YZX.dims == 'YZX', "should be YZX"
    assert dc_YZX.shape == (4, 3, 2), "should be (4, 3, 2)"

    #== bogus request should fail
    try:
        dc_YZX = dc_ZYX.reorder_dims('YZXC')
        err = False
    except:
        err = True

    assert err, "bogus request should trigger error"


    if verbose:
        print('-------------------')
        dc_XYZ.about()
        print('-------------------')
        dc_ZYX.about()
        print('-------------------')

    # shape = [5, 4, 1, 20, 6]
    # opts = dict(dims='TZCYX')
    # data = np.arange(np.product(shape)).reshape(shape)
    # dc = DataChunk(data=data, **opts)
    # chk2 = dc.reorder_dims('YXCZT')

    return 0


def test_datachunk_subchunks():
    """testing subchunks (and squeeze)"""
    shape = [1, 2, 1, 4, 3]
    opts = dict(dims='TZCYX')
    data = np.arange(np.product(shape)).reshape(shape)
    dc = DataChunk(data=data, **opts)

    #== pull out one XY frame
    req = dict(Z=1)
    ans = np.asarray([[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]])
    sc0 = dc.subchunk(req=req)
    assert np.all(ans == sc0.data), "subchunk error sc0"

    #== pull out a (partial) XY frame
    req = dict(Y=[1, 2], X=(0, 3), Z=1, T=None, C=[0])
    ans = np.asarray([[15, 16, 17], [18, 19, 20]])
    sc1 = dc.subchunk(req=req)
    assert np.all(ans == sc1.data), "subchunk error sc1"

    #== pull out a smaller 2x2 XY frame
    req = dict(Y=[0, 1], X=(0, 2), Z=0, T=None, C=[0])
    ans = np.asarray([[15, 16], [18, 19]])
    sc2 = sc1.subchunk(req=req)
    assert np.all(ans == sc2.data), "subchunk error sc2"

    #== squeeze to 2D
    sc3 = sc2.squeeze()
    assert np.all(sc3.shape == (2, 2)), "subchunk squeeze error"
    assert np.all(sc3.dims == ['Y', 'X']), "subchunk squeeze error"

    return 0


def test_datachunk_json(verbose=False):
    """testing conversion to/from a dictionary and to/from json file"""

    shape = [1, 2, 1, 4, 3]
    opts = dict(dims='TZCYX')
    data = np.arange(np.product(shape), dtype=np.uint8).reshape(shape)
    dc_A = DataChunk(data=data, **opts)

    #======================================================
    #   Convert a DataChunk to and from a jdict (json compatible dictionary)
    #       dc_A -> jdict -> dc_B
    dc_B = DataChunk.from_jdict(dc_A.to_jdict())
    hashA = hash(json.dumps(dc_A.to_jdict()))
    hashB = hash(json.dumps(dc_B.to_jdict()))
    assert hashA == hashB, "dictionaries should be equal"

    #======================================================
    #   Convert a DataChunk to and from a json file
    #       dc_A -> chunk.json -> dc_B

    fldr = os.path.join(os.path.dirname(__file__), 'scratch')

    #== raw data encoding
    jf = os.path.join(fldr, 'chunk-raw.json')
    dc_A.to_json(jf, data_encoding='raw')
    dc_B = DataChunk.from_json(jf)

    hashA = hash(json.dumps(dc_A.to_jdict()))
    hashB = hash(json.dumps(dc_B.to_jdict()))
    assert hashA == hashB, "dictionaries should be equal"


    #== base64 data encoding
    jf = os.path.join(fldr, 'chunk-b64str.json')
    dc_A.to_json(jf, data_encoding='b64str')
    dc_B = DataChunk.from_json(jf)

    hashA = hash(json.dumps(dc_A.to_jdict()))
    hashB = hash(json.dumps(dc_B.to_jdict()))
    assert hashA == hashB, "dictionaries should be equal"

    return 0


if __name__ == '__main__':



    print('-----------------------')
    print('test_datachunk_montage')
    test_datachunk_montage()
    print('  PASS test_datachunk_montage')
    print('-----------------------')



    print('-----------------------')
    print('test_datachunk_reorder_dims')
    test_datachunk_reorder_dims()
    print('  PASS test_datachunk_reorder_dims')
    print('-----------------------')

    print('-----------------------')
    print('test_datachunk_subchunks')
    test_datachunk_subchunks()
    print('  PASS test_datachunk_subchunks')
    print('-----------------------')

    print('-----------------------')
    print('test_datachunk_json')
    test_datachunk_json()
    print('  PASS test_datachunk_json')
    print('-----------------------')


