import math
import geopandas as gpd
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import pickle


def get_conditions(inp, con):
    """

    :param inp: a number or something else
    :param con: a condition in form of ">n" etc.
    :return: True or False
    """
    if con[0] == ">" and con[1] != "=":
        return inp > float(con[1:])
    elif con[0] == "<" and con[1] != "=":
        return inp < float(con[1:])
    elif con[0] == ">" and con[1] == "=":
        return inp >= float(con[2:])
    elif con[0] == "<" and con[1] == "=":
        return inp <= float(con[2:])


def extract_by_value(file, save, *value):
    """

    :param file: input image file
    :param save: output image file
    :param value: the value to be extracted, it's type can be num or string
        Can receive a mix of this two type of value
        ## a legal string input for value can be ">80", "<=86.5" etc. ##
        ## but the string input should be no more than a couple of ">" and "<" ##
    :return: None
    """
    in_r = gdal.Open(file)
    driver = in_r.GetDriver()
    x, y = (in_r.RasterXSize, in_r.RasterYSize)
    proj = in_r.GetProjection()
    tran = in_r.GetGeoTransform()
    typ = in_r.GetRasterBand(1).DataType
    or_data = in_r.GetRasterBand(1).ReadAsArray()

    new_data = or_data.copy()
    num_con = []
    str_con = []
    for con in value:
        if type(con) == str:
            str_con.append(con)
        else:
            num_con.append(con)

    for i in range(y):
        for j in range(x):
            flag2 = flag1 = None
            if len(num_con) != 0:
                flag1 = new_data[i, j] in num_con
            if len(str_con) != 0:
                flag2 = get_conditions(new_data[i, j], str_con[0])
                for con in str_con[1:]:
                    flag2 = (flag2 and get_conditions(new_data[i, j], con))
            if not (flag1 or flag2):
                new_data[i, j] = 0

    dts = driver.Create(
        save,
        xsize=x,
        ysize=y,
        bands=1,
        eType=typ,
    )

    dts.SetGeoTransform(tran)
    dts.SetProjection(proj)
    dts.GetRasterBand(1).WriteArray(new_data)
    dts.GetRasterBand(1).SetNoDataValue(0)


def make_grid(loc_in, step, loc_out):
    # 统一坐标系
    crs_file = open("crs.txt", "rb")
    crs = pickle.load(crs_file)
    crs_file.close()
    shp = gpd.read_file(loc_in).to_crs(crs)
    # 确定范围
    bounds = shp.bounds
    bounds_min = bounds.min()
    bounds_max = bounds.max()
    min_x = float(bounds_min.minx)
    max_x = float(bounds_max.maxx)
    min_y = float(bounds_min.miny)
    max_y = float(bounds_max.maxy)

    x, y = (min_x, max_y)

    # 生成网格
    # 点顺序固定 左上-右上-右下-左下
    step = step
    geom_array = []
    while y >= min_y:
        while x <= max_x:
            geom = geometry.Polygon([(x, y), (x+step, y), (x+step, y-step), (x, y-step), (x, y)])
            geom_array.append(geom)
            x += step
        x = min_x
        y -= step

    # 显示
    grid = gpd.GeoDataFrame(geometry=geom_array, crs=crs)
    ind = np.stack([grid.intersects(shp.geometry[0]).values,
                    grid.intersects(shp.geometry[1]).values,
                    grid.intersects(shp.geometry[2]).values]).any(0)
    # 裁剪网格
    grid = grid.loc[ind]
    fig, ax = plt.subplots(1, 2)
    ax[0] = shp.plot(ax=ax[0])
    ax[1] = grid.plot(ax=ax[1])
    plt.show()
    grid.to_file(loc_out)
    return grid


def guss(threshold, dist):
    q1 = math.exp(-(1/2.0)*(dist/threshold)*(dist/threshold))
    q2 = math.exp(-(1/2.0))
    return (q1 - q2)/(1 - q2)


def guss_reach(threshold, pop_grid, area_grid):
    """
    :param threshold: the threshold of guss search
    :param pop_grid: grid of popularity
    :param area_grid: grid of area of research region
    :return: a dic that contain a guss search value of specific gird in pop_grid
    warning : the input grid should be in the same crs and have same area of gird
    """


class Zonal:
    def __init__(self, in_raster, in_shapefile):
        # read information of raster inputted
        self.ras = gdal.Open(in_raster)
        self.trans = self.ras.GetGeoTransform()
        self.size = (self.ras.RasterXSize, self.ras.RasterYSize)
        self.start_of_x, self.start_of_y = (self.trans[0], self.trans[3])
        self.cell_size_x, self.cell_size_y = (self.trans[1], self.trans[5])
        self.x_max, self.y_min = (self.start_of_x + self.size[0] * self.cell_size_x,
                                  self.start_of_y + self.size[1] * self.cell_size_y)
        self.array = self.ras.GetRasterBand(1).ReadAsArray()
        # print(array)
        self.ras = None

        # read information of shapefile inputted
        self.shp = gpd.read_file(in_shapefile)
        self.crs = self.shp.crs

        # left then right, up then down
        # the value here will be changed in each time when count_by_grid are running
        self.offsets_x_left, self.offsets_x_right = 0, 0
        self.offsets_y_up, self.offsets_y_down = 0, 0
        self.covered_x_start, self.covered_y_start = 0, 0
        self.covered_x_end, self.covered_y_end = 0, 0

    def grid_contain(self, m, n):
        contain_param = 0
        if m == self.covered_y_start:
            if n == self.covered_x_start:
                contain_param = ((abs(self.cell_size_y) - self.offsets_y_up) * (self.cell_size_x - self.offsets_x_left))
            elif n == self.covered_x_end:
                contain_param += ((abs(self.cell_size_y) - self.offsets_y_up) * self.offsets_x_right)
            else:
                contain_param += (self.cell_size_x * self.offsets_y_up)
        elif m == self.covered_y_end:
            if n == self.covered_x_start:
                contain_param += (self.offsets_y_up * (self.cell_size_x - self.offsets_x_left))
            elif n == self.covered_x_end:
                contain_param += (self.offsets_y_down * self.offsets_x_right)
            else:
                contain_param += (self.offsets_y_down * self.cell_size_x)
        elif n == self.covered_x_start:
            contain_param += ((self.cell_size_x - self.offsets_x_left) * abs(self.cell_size_y))
        elif n == self.covered_x_end:
            contain_param += (self.offsets_x_right * abs(self.cell_size_y))
        else:
            contain_param += self.cell_size_x * abs(self.cell_size_y)

        # print(contain_param)
        return contain_param

    def count_by_grid(self, typ):
        result = {}
        count = 0
        for grid_polygon in self.shp["geometry"]:
            p_x, p_y = grid_polygon.exterior.coords.xy
            # the order should be left-up -> right-up -> right-down -> left-down
            for i in range(len(p_x)-1):

                if p_x[i] > self.x_max:
                    p_x = self.x_max
                if p_y[i] < self.y_min:
                    p_y = self.y_min
                point = (p_x[i], p_y[i])

                closest_x = math.floor((point[0] - self.start_of_x) / self.cell_size_x)
                closest_y = math.floor((point[1] - self.start_of_y) / self.cell_size_y)
                # print(closest_x, closest_y)

                if i == 0:
                    self.covered_x_start, self.covered_y_start = closest_x, closest_y
                    # the offset is always the distance to left or up
                    # left , up offset
                    self.offsets_x_left = abs(point[0] - self.start_of_x - self.cell_size_x * closest_x)
                    self.offsets_y_up = abs(point[1] - self.start_of_y - self.cell_size_y * closest_y)
                if i == 2:
                    self.covered_x_end, self.covered_y_end = closest_x, closest_y
                    # the offset is always the distance to left or up
                    # right , down offset
                    self.offsets_x_right = abs(point[0] - self.start_of_x - self.cell_size_x * closest_x)
                    self.offsets_y_down = abs(point[1] - self.start_of_y - self.cell_size_y * closest_y)

            # print(offsets_x, offsets_y)
            # print(covered_x_start, covered_x_end, ";", covered_y_start, covered_y_end)
            # print(offsets_x_left, offsets_x_right, ";", offsets_y_up, offsets_y_down)
            # calculate the area that cover the pixel having legal value
            if typ == "area":
                area = 0
                for n in range(self.covered_x_start, self.covered_x_end):
                    for m in range(self.covered_y_start, self.covered_y_end):
                        flag = 0
                        if self.array[m][n] > 0:
                            flag = 1
                        # F**king duplicate, I won't rewrite it to a class, never.
                        # seal a function would need above 10 param!
                        area += self.grid_contain(m, n) * flag

                result[count] = area
            if typ == "sum":
                sums = 0
                for n in range(self.covered_x_start, self.covered_x_end):
                    for m in range(self.covered_y_start, self.covered_y_end):
                        flag = 0
                        if self.array[m][n] > 0:
                            flag = self.array[m][n]/abs(self.cell_size_y*self.cell_size_x)
                        sums += self.grid_contain(m, n) * flag

                result[count] = sums
            # mind THIS F**king terrible tab plz !!
            count += 1

        # bid = list(result.keys())
        # result = pd.DataFrame(result.values(), columns=[typ])
        # result["BID"] = bid
        self.shp[typ] = result.values()
        # print(result)
        return self.shp


# extract_by_value("Raster/2000t.tif", "test.tif", ">=10", "<=50")
# g = make_grid("ShapeFile/T.shp", 1000, "Rasterouttest/grid.shp")
zn = Zonal("Raster/2000t.tif", "Rasterouttest/grid.shp")
re = zn.count_by_grid("area")
print(re)
