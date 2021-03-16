#!/usr/bin/python

from math import sqrt
import sys
import argparse
from collections import deque
from struct import pack, unpack
import numpy as np

from osgeo import gdal
import math
import decimal


from concavehull import ConcaveHull

gdal.UseExceptions()
gdal.TermProgress = gdal.TermProgress_nocb


#
# NormalVector
#
# Calculate the normal vector of a triangle. (Unit vector perpendicular to
# triangle surface, pointing away from the "outer" face of the surface.)
# Computed using 32-bit float operations for consistency with other tools.
#
# Parameters:
#  triangle vertices (nested x y z tuples)
#
# Returns:
#  normal vector (x y z tuple)
#
def NormalVector(t):
    (ax, ay, az) = t[0]
    (bx, by, bz) = t[1]
    (cx, cy, cz) = t[2]

    # first edge
    e1x = np.float32(ax) - np.float32(bx)
    e1y = np.float32(ay) - np.float32(by)
    e1z = np.float32(az) - np.float32(bz)

    # second edge
    e2x = np.float32(bx) - np.float32(cx)
    e2y = np.float32(by) - np.float32(cy)
    e2z = np.float32(bz) - np.float32(cz)

    # cross product
    cpx = np.float32(e1y * e2z) - np.float32(e1z * e2y)
    cpy = np.float32(e1z * e2x) - np.float32(e1x * e2z)
    cpz = np.float32(e1x * e2y) - np.float32(e1y * e2x)

    # return cross product vector normalized to unit length
    mag = np.sqrt(np.power(cpx, 2) + np.power(cpy, 2) + np.power(cpz, 2))
    return cpx / mag, cpy / mag, cpz / mag


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = decimal.Decimal(math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1])))
    b = decimal.Decimal(math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1])))
    c = decimal.Decimal(math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1])))
    # print(point_1, point_2, point_3, a, b, c)
    A = math.degrees(math.acos(decimal.Decimal((a * a - b * b - c * c) / (-2 * b * c))))
    B = math.degrees(math.acos(decimal.Decimal((b * b - a * a - c * c) / (-2 * a * c))))
    # print((c * c - a * a - b * b) / (-2 * a * b))
    C = math.degrees(math.acos(decimal.Decimal((c * c - a * a - b * b) / (-2 * a * b))))

    return [A, B, C]


# stlwriter is a simple class for writing binary STL meshes.
# Class instances are constructed with a predicted face count.
# The output file header is overwritten upon completion with
# the actual face count.
class stlwriter():

    # path: output binary stl file path
    # facet_count: predicted number of facets
    def __init__(self, path, facet_count=0):
        self.f = open(path, 'wb')

        # track number of facets actually written
        self.written = 0

        # write binary stl header with predicted facet count
        self.f.write(str.encode('\0' * 80))
        # (facet count is little endian 4 byte unsigned int)
        self.f.write(pack('<I', facet_count))

    # t: ((ax, ay, az), (bx, by, bz), (cx, cy, cz))
    def add_facet(self, t):
        # facet normals and vectors are little endian 4 byte float triplets
        # strictly speaking, we don't need to compute NormalVector,
        # as other tools could be used to update the output mesh.
        self.f.write(pack('<3f', *NormalVector(t)))
        for vertex in t:
            self.f.write(pack('<3f', *vertex))
        # facet records conclude with two null bytes (unused "attributes")
        self.f.write(str.encode('\0\0'))
        self.written += 1

    def done(self):
        # update final facet count in header before closing file
        self.f.seek(80)
        self.f.write(pack('<I', self.written))
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done()


def fail(msg):
    print(msg)
    exit(1)


def log(msg):
    if args.verbose:
        print(msg)


ap = argparse.ArgumentParser(description='Convert a GDAL raster (like a GeoTIFF heightmap) to an STL terrain surface.')
ap.add_argument('-x', action='store', default=0.0, type=float, help='Fit output x to extent (mm)')
ap.add_argument('-y', action='store', default=0.0, type=float, help='Fit output y to extent (mm)')
optg_z = ap.add_mutually_exclusive_group()
optg_z.add_argument('-z', action='store', default=None, type=float,
                    help='Z scale expressed as a vertical scale factor (1)')
optg_z.add_argument('-s', action='store', default=None, type=float,
                    help='Z scale expressed as a ratio of vertical units per horizontal unit (1)')
ap.add_argument('-b', '--base', action='store', default=0.0, type=float, help='Base height (0)')
ap.add_argument('-c', '--clip', action='store_true', default=False, help='Clip z to minimum elevation')
ap.add_argument('-v', '--verbose', action='store_true', default=False, help='Print log messages')
ap.add_argument('--band', action='store', default=1, type=int, help='Raster data band (1)')
ap.add_argument('-m', '--minimum', action='store', default=None, type=float,
                help='Omit vertices below minimum elevation')
ap.add_argument('-M', '--maximum', action='store', default=None, type=float,
                help='Omit vertices above maximum elevation')
optg_region = ap.add_mutually_exclusive_group()
optg_region.add_argument('-w', '--window', action='store', default=None, type=float, nargs=4,
                         help='Opposing corner coordinates in geographic CRS')
optg_region.add_argument('-p', '--pixels', action='store', default=None, type=float, nargs=4,
                         help='Opposing corner coordinates in pixel coordinates')
ap.add_argument('RASTER', help='Input heightmap image')
ap.add_argument('STL', help='Output STL path')
args = ap.parse_args()

try:
    img = gdal.Open(args.RASTER)
except RuntimeError as e:
    fail(str(e).strip())

# input raster dimensions
w = img.RasterXSize
h = img.RasterYSize
log("raster dimensions = (%s, %s)" % (str(w), str(h)))

# get default transformation from image coordinates to world coordinates
# note that since we obtain this transformation before making any region
# selection, and since this transformation is used for output, subset
# window regions will be aligned correctly in output stl coordinates.
t = img.GetGeoTransform()

git = gdal.InvGeoTransform(t)[1]
log("geo->pixel transform: %s" % str(git))

if args.window is not None:
    # if a geographic window is specified, convert it to a pixel window in input raster coordinates

    # apply inverse geo transform to window points
    px0, py0 = gdal.ApplyGeoTransform(git, args.window[0], args.window[1])
    px1, py1 = gdal.ApplyGeoTransform(git, args.window[2], args.window[3])

    # set arg.pixels to obtained pixel points
    args.pixels = [px0, py0, px1, py1]

if args.pixels is None:
    # if no pixel extent window is specified, use whole input raster.
    xmin = 0
    ymin = 0
    ww = w
    wh = h
else:
    # if a pixel extent window is specified (either directly with --pixels or
    # derived from a geographic --window), clip it to the input raster extent.

    xmin = int(round(min(args.pixels[0], args.pixels[2])))
    ymin = int(round(min(args.pixels[1], args.pixels[3])))

    xmax = int(round(max(args.pixels[0], args.pixels[2])))
    ymax = int(round(max(args.pixels[1], args.pixels[3])))

    if xmin >= w:
        fail("Region of interest lies entirely outside raster (xmin)")

    if ymin >= h:
        fail("Region of interest lies entirely outside raster (ymin")

    if xmax <= 0:
        fail("Region of interest lies entirely outside raster (xmax)")

    if ymax <= 0:
        fail("Region of interest lies entirely outside raster (ymax)")

    # if we passed those tests, at least part of the window overlaps the raster,
    # so we can safely clip to the raster extent and still have something

    if xmin < 0:
        xmin = 0

    if ymin < 0:
        ymin = 0

    if xmax > w:
        xmax = w

    if ymax > h:
        ymax = h

    ww = xmax - xmin
    wh = ymax - ymin

log("xmin, ymin = %d, %d" % (xmin, ymin))
log("ww, wh = %d, %d" % (ww, wh))

# output mesh dimensions are one row and column less than raster window
mw = ww - 1
mh = wh - 1

# save x pixel size if needed for scaling
xyres = abs(t[1])

# Apply z scale factor, if any
if args.z is not None:
    zscale = args.z
elif args.s is not None:
    zscale = 1.0 / args.s
else:
    zscale = 1.0

# recalculate z scale and xy transform if different dimensions are requested
if args.x != 0.0 or args.y != 0.0:

    # recaculate xy scale based on requested x or y dimension
    # if both x and y dimension are set, select smaller scale
    if args.x != 0.0 and args.y != 0.0:
        pixel_scale = min(args.x / mw, args.y / mh)
    elif args.x != 0.0:
        pixel_scale = args.x / mw
    elif args.y != 0.0:
        pixel_scale = args.y / mh

    # adjust z scale to maintain proportions with new xy scale
    zscale *= pixel_scale / xyres

    # revise transformation matrix
    # image: 0,0 at top left corner of top left pixel (0.5,0.5 at pixel center)
    t = (
        -pixel_scale * mw / 2.0,  # 0 left edge of top left pixel
        pixel_scale,  # 1 pixel width
        0,  # 2
        pixel_scale * mh / 2.0,  # 3 top edge of top left pixel
        0,  # 4
        -pixel_scale  # 5 pixel height
    )

log("transform = %s" % str(t))

band = img.GetRasterBand(args.band)
nd = band.GetNoDataValue()

# map GDAL pixel data type to corresponding struct format character
typemap = {
    gdal.GDT_Byte: 'B',
    gdal.GDT_UInt16: 'H',
    gdal.GDT_Int16: 'h',
    gdal.GDT_UInt32: 'I',
    gdal.GDT_Int32: 'i',
    gdal.GDT_Float32: 'f',
    gdal.GDT_Float64: 'd'
}

typeName = gdal.GetDataTypeName(band.DataType)
if band.DataType not in typemap:
    fail('Unsupported data type: %s' % typeName)

# rowformat is used to unpack a row of raw image data to numeric form
rowformat = typemap.get(band.DataType) * ww
log("data type = %s" % typeName)
log("type format = %s" % typemap.get(band.DataType))

# min, max, mean, sd; min used for z clipping
stats = band.GetStatistics(True, True)
log("min, max, mean, sd = %s" % str(stats))

# zmin is subtracted from elevation values
if args.clip == True:
    zmin = stats[0]
    zmax = stats[1]
else:
    zmin = stats[0]
    zmax = stats[1]
log("zmin = %s, zmax= %s" % (str(zmin), str(zmax)))

# Rolling pixel buffer has space for two rows of image data.
# Old data is automatically discarded as new data is loaded.
pixels = deque(maxlen=(2 * ww))
log("buffer size = %s" % str(pixels.maxlen))

# Initialize pixel buffer with first row of data from the image window.
pixels.extend(unpack(rowformat, band.ReadRaster(xmin, ymin, ww, 1, ww, 1, band.DataType)))

# Precalculate output mesh size (STL is 50 bytes/facet + 84 byte header)
# Actual facet count and file size may differ (be less) if pixels are skipped as nodata or out of range.
facetcount = mw * mh * 2
filesize = (facetcount * 50) + 84
log("predicted (max) facet count = %s" % str(facetcount))
log("predicted (max) STL file size = %s bytes" % str(filesize))


# skip(v) tests if elevation value v should be omitted from output.
# It returns True if v matches the nodata value, if v is less than
# the minimum allowed elevation, or if v is greater than the
# maximum allowed elevation. Otherwise it returns False.
def skip(v):
    global nd
    global args
    if v == nd:
        return True
    if args.minimum is not None and v < args.minimum:
        return True
    if args.maximum is not None and v > args.maximum:
        return True
    return False


points_3d = []

for y in range(mh):
    pixels.extend(unpack(rowformat, band.ReadRaster(xmin, ymin + y + 1, ww, 1, ww, 1, band.DataType)))
    for x in range(mw):
        av = pixels[x]

        if skip(av):
            continue

        points_3d.append([t[0] + ((xmin + x) * t[1]) + ((ymin + y) * t[2]),
                          t[3] + ((xmin + x) * t[4]) + ((ymin + y) * t[5]),
                          zscale * float(av)])

points_3d = np.asarray(points_3d)
points_2d = points_3d[:, :2]

from scipy.spatial import Delaunay

tri = Delaunay(points_2d)

mask = []
for simple in tri.simplices:
    in_degrees = cal_ang(points_2d[simple[0]], points_2d[simple[1]], points_2d[simple[2]])
    dmax = max(in_degrees)
    # dind = in_degrees.index(dmax)
    mask.append(dmax <= 90.0001)

with stlwriter(args.STL, facetcount) as mesh:
    for simple in tri.simplices[mask]:
        a = points_3d[simple[0]]
        b = points_3d[simple[1]]
        c = points_3d[simple[2]]
        mesh.add_facet((a, b, c))
        # print(a, b, c)

    for simple in tri.simplices[mask]:
        a1 = [points_3d[simple[0]][0], points_3d[simple[0]][1], points_3d[simple[0]][2]]
        a1[2] = zscale * zmin + args.base
        b1 = [points_3d[simple[1]][0], points_3d[simple[1]][1], points_3d[simple[1]][2]]
        b1[2] = zscale * zmin + args.base
        c1 = [points_3d[simple[2]][0], points_3d[simple[2]][1], points_3d[simple[2]][2]]
        c1[2] = zscale * zmin + args.base
        mesh.add_facet((a1, b1, c1))
        # print(a1, b1, c1)

    tri.simplices = tri.simplices[mask]

    ch = ConcaveHull()

    ch.loadpoints(points_2d)
    ch.calculatehull()
    hull_points = np.vstack(ch.boundary.exterior.coords.xy).T.tolist()
    # print(hull_points[:2])

    points_2d = points_2d.tolist()
    # print(points_2d[:2])
    for i, hp in enumerate(hull_points):
        if i == len(hull_points)-1:
            continue
        # print(hp)
        ind = points_2d.index(hp)
        a = (hp[0], hp[1], points_3d[ind][2])
        a1 = (hp[0], hp[1], zscale * zmin + args.base)
        hp_1 = hull_points[i+1]
        ind = points_2d.index(hp_1)
        b = (hp_1[0], hp_1[1], points_3d[ind][2])
        b1 = (hp_1[0], hp_1[1], zscale * zmin + args.base)

        # print(a, a1, b, b1)
        mesh.add_facet((a, a1, b))
        mesh.add_facet((b, b1, a1))
    # log("actual facet count: %s" % str(mesh.written))
