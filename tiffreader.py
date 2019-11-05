import itertools
import pdb
import os

import tifffile as tf
import numpy as np
import pandas as pd

import datachunk as dc
#import bfdataarray


class TiffMetadataParser(object):
    def __init__(self, f=None):
        raise Exception('TiffMetadataParser DNE, parsing is handled by the TiffReader')



class TiffReader(object):
    """for pulling frame(s) and chunks out of a tiff hyperstack

    TODO: lil-tiff (first N frames to a new file, allows fiji inspection
          to determine metadata parameters (dim order, num-Z, starting frame
          etc...))
    """

    def __init__(self, f=None):

        if isinstance(f, list):
            self.files = f
        else:
            self.files = [f]

        self.TF = [tf.TiffFile(f) for f in self.files]
        self.build_index()

        #self.mmmd = [tiff.micromanager_metadata for tiff in self.TF]
        return

    def build_index(self):
        """global frame index for >=1 tiff file(s) and metadata

        The dataframe df holds all of the tiff page indexing info. Columns
        T, Z, and C are self-explanatory. F is the (input) file index and 'ndx'
        is the page index WITHIN the corresponding file. Given, TZC, one can
        determine F and ndx.

        tzc2fndx is a dictionary. Keys are (T,Z,C) tuples (converted to strings)
        and values are (F, ndx) tuples

        """

        # extract some info from zeroth series of zeroth file
        series0 = self.TF[0].series[0]
        axes = series0.axes
        shape = series0.shape
        dtype = series0.dtype

        if axes[-2:] not in ['XY', 'YX']:
            raise Exception('expecting the last two axes to be XY or YX ')

        # split axes into tzc and xy
        xy_axes = axes[-2:]
        xy_shape = shape[-2:]
        tzc_axes = axes[:-2]
        tzc_shape = shape[:-2]
        tzc_ndx = list(itertools.product(*[range(n) for n in tzc_shape]))

        # a dataframe to hold all the indexing information
        df = pd.DataFrame(data=tzc_ndx, columns=list(tzc_axes))
        for col in ['T', 'Z', 'C']:
            if col not in df.columns:
                df[col] = np.zeros(len(df), dtype=int)

        filesinfo = []
        fndx = []
        ndx = []
        for i, ff in enumerate(self.files):
            tiff = self.TF[i]
            num_pages = len(tiff.pages)
            fndx += [i]*num_pages
            ndx += list(range(num_pages))
            filesinfo.append(dict(f=ff, num_pages=num_pages))
            
        if len(ndx) != len(df):
            print('WARNING: something is wrong')
            raise Exception()


        df['F'] = fndx
        df['ndx'] = ndx
        df = df[['T', 'Z', 'C', 'F', 'ndx']]

        # given TZC, return [file,frame] indices
        tzc2fndx = {str(tuple(x[:3])): (x[3],x[4]) for x in df.values}
        #tzc2ndx = {str(tuple(x[:3])): x[3],x[4]) for x in df.values}

        meta = dict(
            _about="tiff metadata",
            files=self.files,
            filesinfo=filesinfo,
            shape=shape,
            axes=axes,
            dtype=dtype,
            axes_xy=xy_axes,
            shape_xy=xy_shape,
            axes_tzc=tzc_axes,
            shape_tzc=tzc_shape,
            df_tzc=df,
            tzc2fndx=tzc2fndx
            )

        self.meta = meta

        return


    def about(self):
        """helpful information at a glance"""
        print('#------------------------')
        print('# TiffReader metadata')
        print('#------------------------')
        print('num_files :', len(self.meta['files']))
        for i, ff in enumerate(self.meta['filesinfo']):
            print('file%3.3i   :' % i, ff['f'])
        print('pages/file:', [ff['num_pages'] for ff in self.meta['filesinfo']])
        print('dtype     :', self.meta['dtype'])
        print('axes      :', self.meta['axes'])
        print('shape     :', self.meta['shape'])
        print('df_tzc    :')
        print(pd.concat([self.meta['df_tzc'].head(), self.meta['df_tzc'].tail()]))
        print()
        return


    def getframe(self, Z=0, T=0, C=0):
        """returns a (YX) frame given (hyper)stack indices Z,T,C

        TODO: turbo request of a frame subregion?
        TODO: what other metadata to carry forward?
        TODO: return a datachunk?
        """
        #index = self.meta['tzc2ndx'][str(tuple([T, Z, C]))]
        #data = self.tiff.pages[index].asarray()

        f, index = self.meta['tzc2fndx'][str(tuple([T, Z, C]))]
        data = self.TF[f].pages[index].asarray()

        axes = self.meta['axes_xy']
        shape = self.meta['shape_xy']
        dtype = self.meta['dtype']
        meta = dict(Z=Z, T=T, C=C, dtype=dtype)

        frame = dict(data=data, axes=axes, shape=shape, meta=meta)
        return frame

    def getchunk(self, req=None):
        """get multiple frames and assemble a DataChunk

        TODO: better naming for chunk requests (compact vs full?)
        """
        dtype = self.meta['dtype']

        #== confirm that requested axes exist in the file
        for a in req.keys():
            if a not in self.meta['axes']:
                raise Exception('requested axis (%s) not in the tiff axes (%s)'
                % (a, self.meta['axes']))

        #===========================================================
        #== enumerate TZC combinations (frames) and generate ALL
        #== of the getframe requests. Each request is a dictionary
        #===========================================================
        shape_tzc = self.meta['shape_tzc']
        axes_tzc = self.meta['axes_tzc']
        req_tzc = {k:req.get(k, None) for k in axes_tzc}
        ix_tzc = dc.chunk_ix(shape=shape_tzc, dims=axes_tzc, req=req_tzc)
        #== list of tuples
        TZC = list(itertools.product(*ix_tzc))

        #== the XY request (basically a crop) is done on each TZC frame
        shape_xy = self.meta['shape_xy']
        axes_xy = self.meta['axes_xy']
        req_xy = {k:req.get(k, None) for k in 'XY'}
        ix_xy = dc.chunk_ix(shape=shape_xy, dims=axes_xy, req=req_xy)

        #== determine the shape of the output chunk
        new_shape_tzc = [len(x) for x in ix_tzc]
        new_shape_xy = [len(x) for x in ix_xy]

        #== build a list of frames, then reshape it
        shp = [len(TZC)] + new_shape_xy
        data = np.zeros(shp, dtype=dtype)
        for i, row in enumerate(TZC):
            reqi = dict(zip(axes_tzc, row))
            frm = self.getframe(**reqi)
            # if no req_xy, do not make a DataChunk
            if req_xy['X'] == None and req_xy['Y'] == None:
                data[i] = frm['data']
            else:
                frm['dims'] = frm['axes']
                frm.pop('axes')
                chk = dc.DataChunk(**frm).subchunk(req=req_xy)
                data[i] = chk.data
            #print(row, dict(zip(axes_tzc, row)))

        #== reshape
        shape = new_shape_tzc+new_shape_xy
        data = data.reshape(shape)

        #== build the DataChunk
        axes = axes_tzc + axes_xy
        meta = dict(_about='DataChunk from a tiff file',
                    req=req)
        chunk = dc.DataChunk(data=data, dims=axes, meta=meta)

        return chunk


    #def to_bfda(self):
        """convert to a bfd BFDataArray"""

        #da = self.to_xrda()
        #bfdarr = bfdataarray.BFDataArray(da=da,)
        #return bfdarr


    def to_xrda(self, coords=None):
        """convert the tiff to an xarray.DataArray

        coords are invented
        """
        import xarray as xr

        data = self.tiff.asarray()
        dims = self.meta['axes']
        shape = self.meta['shape']
        dims = [a for a in axes]

        attrs = dict(_about='created using TiffReader.to_xrda()')

        if coords is None:
            #== pixel coords (cuz no more info provided)
            coords = {}
            for ax, sz in zip(dims, shape):
                coords[ax] = range(sz)
                if ax == 'C' and sz == 3:
                    coords[ax] = ['r', 'g', 'b']
                if ax == 'C' and sz > 3:
                    #raise Exception('implement >3 channels')
                    coords[ax] = ['r', 'g', 'b']
                    coords[ax] += ['ch%2.2i' % i for i in range(3,sz)]



        xrd = xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)

        return xrd

class TiffInspector(object):
    """exploring the metadata morass"""

    def __init__(self, f=None):

        self.filename = f
        self.tiff = tf.TiffFile(f)
    
        self.topdict = self.tiff.__dict__
        self.pg0dict = self.tiff.pages[0].__dict__
        self.sr0dict = self.tiff.series[0].__dict__
        self.imjdict = self.topdict.get('imagej_metadata', {})

        self.topkeys = sorted(self.topdict.keys())
        self.pg0keys = sorted(self.pg0dict.keys())
        self.sr0keys = sorted(self.sr0dict.keys())
        self.imjkeys = sorted(self.imjdict.keys())

        mmmd = self.tiff.micromanager_metadata

        return

    def contents(self):
        """summarize the tiff contents

        about TiffFile attributes
        pages : list of TiffPage
            All TIFF pages in file. (more basic than series)
        series : list of TiffPageSeries
            TIFF pages with compatible shapes and types.
        """

        print('--------------------------------')
        print('topdict (TiffFile object) contents:')
        for k in self.topkeys:
            print('%20s :' % k, type(self.topdict[k]))

        print('--------------------------------')
        print('pg0dict  (page[0]) contents:')
        for k in self.pg0keys:
            print('%20s :' % k, type(self.pg0dict[k]))

        print('--------------------------------')
        print('sr0dict (series[0]) contents:')
        for k in self.sr0keys:
            print('%20s :' % k, type(self.sr0dict[k]))

        print('--------------------------------')
        print('imagej_metadata (from topdict) contents:')
        for k in self.imjkeys:
            print('%20s :' % k, type(self.imjdict[k]))


        print('--------------------------------')

        return

    def parse_imj(self):
        """specific to the imagej_metadata

        NOTE: The 'Info' entry might be an OME-XML data block, or nothing
        NOTE: under construction, only prints some info
        """

        keys = ['images',
                'slices',
                'frames',
                'channels',
                'unit',
                'hyperstack',
                'loop',
                'ImageJ',
                'Ranges',
                'min',
                'max',
                'mode'
                ]

        metadata = self.imjdict

        print('------------------------------------')
        print('   imagej_metadata')
        print('------------------------------------')

        #== big xml string (with things like PhysicalSizeX)
        info = metadata.get('Info', None)
        if info is not None:
            print('Info:', info[:800])   #== very big xml (string)

        for k in keys:
            print('%20s :' % k, metadata.get(k, None))

        ranges = []
        if 'Ranges' in metadata.keys():
            ranges = list(metadata['Ranges'])
        elif 'min' in metadata.keys() and 'max' in metadata.keys():
            ranges.append(metadata['min'])
            ranges.append(metadata['max'])

        return

    def parse_pg0(self):
        """information in page[0]
        
        'description' sometimes has info such as spatial units, maybe redundant
        'tags' can theoretically store useful things..

        NOTE: under construction, returns nothing
        """

        tagkeys = ['XResolution',
                    'YResolution',
                    'ResolutionUnit'
                    ]

        keys = ['shape',
                'is_contiguous',
                'samplesperpixel',
                'description',
                ]

        print('------------------------------------')
        print('   pg0_metadata')
        print('------------------------------------')

        pg0tags = self.pg0dict.get('tags', {})
        tags = sorted(pg0tags.keys())

        for k in tags:
            lbl = 'tags[\'%s\']' % k
            print('%25s :' % lbl, pg0tags.get(k, None))

        for k in keys:
            print('%25s :' % k, self.pg0dict.get(k, None))

        return
    
#==============================================================================
#                            TEST DEMARCATION
#               everything below is tests, above is code
#==============================================================================

def test_toxrda():
    pass


def test00(verbose=False):
    """load demo tiff (TZCYX) with rgb blocks, pull out some chunks"""

    #== path to the demo tiff file
    dn = os.path.dirname(__file__)
    demotiff = '../example_data/test_tiffs/demo_01_rgb_blocks.tiff'
    f = os.path.join(dn, demotiff)

    tr = TiffReader(f=f)

    #==========================================================
    #== request the red box with 1px border (frame 0)
    #== the box pixel values should be 189
    req = dict(T=0, Z=1, C=0, Y=(5,11), X=(4,11))
    chk0 = tr.getchunk(req=req).squeeze()

    row0 = [0]*7
    row1 = [0]+[189]*5+[0]
    row5 = [0]*7
    assert chk0.shape == (6, 7), "should be (6, 7)"
    assert chk0.dims == ['Y', 'X'], "should be ['Y', 'X']"
    assert np.all(chk0.data[0] == row0), "should be zeros"
    assert np.all(chk0.data[1] == row1), "data row error"
    assert np.all(chk0.data[5] == row5), "should be zeros"

    if verbose:
        print('---------------------------------------')
        chk0.about()


    #==========================================================
    #== request timeseries for one pixel in the green box (switching on/off)
    req = dict(Z=1, C=1, Y=10, X=35)
    chk1 = tr.getchunk(req=req).squeeze()

    ans = [0, 190]*10
    assert chk1.shape == (20,), "should be (20,)"
    assert chk1.dims == ['T'], "should be ['T']"
    assert np.all(chk1.data == ans), "data row error"

    if verbose:
        print('---------------------------------------')
        chk1.about()

    #==========================================================
    #== request timeseries for one pixel in the green box (switching on/off)
    #== plus one z-adjacent pixel
    req = dict(T=(0,10), Z=[0,1] ,C=1, Y=10, X=35)
    chk2 = tr.getchunk(req=req).squeeze()

    row0 = [0, 0]
    row1 = [0, 190]
    assert chk2.shape == (10, 2), "should be (10, 2)"
    assert chk2.dims == ['T', 'Z'], "should be ['Z']"
    assert np.all(chk2.data[0] == row0), "data row error"
    assert np.all(chk2.data[1] == row1), "data row error"

    if verbose:
        print('---------------------------------------')
        chk2.about()

    return 0


def test_worm_hyperstack_00(verbose=False):
    """load a single color channel worm hyperstack (TZYX)"""

    #== path to the demo tiff file
    dn = os.path.dirname(__file__)
    demotiff = '../example_data/TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s_01_1-300-hyperstack.ome.tif'
    f = os.path.join(dn, demotiff)

    tr = TiffReader(f=f)

    #== correctly parsed metadata?
    assert tr.meta['shape'] == (25, 12, 512, 146), "should be (25, 12, 512, 146)"
    assert tr.meta['axes'] == 'TZYX', "should be 'TZYX'"

    #== NOTE: under construction
    req = dict(T=(0,10), Z=[0,1] ,Y=10, X=35)
    chk0 = tr.getchunk(req=req)
    chk1 = tr.getchunk(req={})

    tr.about()

    return 0

def test_worm_metadata_00(verbose=False):
    """load tiff metadata"""

    dn = os.path.dirname(__file__)
    #== TZYX worm
    demotiff = '../example_data/TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s_01_1-300-hyperstack.ome.tif'
    f = os.path.join(dn, demotiff)

    tr = TiffReader(f=f)
    #tr.about()

def test_rainbow_worm_00(verbose=False):
    """load a rainbow worm (CZYX)"""

    #== path to the demo tiff file
    dn = os.path.dirname(__file__)
    demotiff = '../example_data/col00-medfilt-3-lowerquantile-0.1upperquant-1e-06-normalize.tif'
    f = os.path.join(dn, demotiff)

    tr = TiffReader(f=f)
    #tr.about()


def test_multiple_tiff_files(verbose=False):
    """multiple tiff files"""


    dn = os.path.dirname(__file__)
    #== TYX worm
    f1 = '../example_data/lite1H20GCaMP6srab3NLSRFPln3_worm2_GFP_2100frames_2_MMStack_Default.ome.tif'
    f2 = '../example_data/lite1H20GCaMP6srab3NLSRFPln3_worm2_GFP_2100frames_2_MMStack_Default_1.ome.tif'
    f1 = os.path.join(dn, f1)
    f2 = os.path.join(dn, f2)

    tr = TiffReader(f=[f1,f2])
    #tr.about()



if __name__ == '__main__':


    print('-----------------------')
    print('test_multiple_tiff_files')
    test_multiple_tiff_files()
    print('  passed test_multiple_tiff_files')
    print('-----------------------')

    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()

    print('-----------------------')
    print('test_worm_metadata_00')
    test_worm_metadata_00()
    print('  passed test_worm_metadata_00')
    print('-----------------------')

    print('-----------------------')
    print('test_worm_hyperstack_00')
    test_worm_hyperstack_00()
    print('  passed test_worm_hyperstack_00')
    print('-----------------------')


    print('-----------------------')
    print('test00')
    test00()
    print('  passed test00')
    print('-----------------------')


    #== path to the demo rainbow worm tiff file
    dn = os.path.dirname(__file__)
    f = os.path.join(dn, '../example_data/col00-medfilt-3-lowerquantile-0.1upperquant-1e-06-normalize.tif')

    if os.path.isfile(f):
        print('-----------------------')
        print('test_rainbow_worm_00')
        test_rainbow_worm_00()
        print('  passed test_rainbow_worm_00')
        print('-----------------------')
