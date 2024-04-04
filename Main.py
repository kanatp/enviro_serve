"""
@Time: 2024/4/4
@author:  Yubing
@purpose: base function to deal with raster/shapefile

"""
import math
import os.path

import geopandas as gpd
import pandas as pd
import shapely
import multiprocessing
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import pickle


class Grid:

    """
    to conveniently use grid shp
    the size was a tuple arrange in form like (size_x, size_y)
    the bound was a list of points the grid's largest circumscribed rectangle has
    """

    def __init__(self, grid_shp, size=None, bound=None):
        self.grid_shp = grid_shp
        if size:
            self.size = size
        else:
            self.size = self.getSize()
        if bound:
            self.bound = bound
        else:
            bounds = self.grid_shp.bounds
            bounds_min = bounds.min()
            bounds_max = bounds.max()
            self.bound = [float(bounds_min.minx), float(bounds_min.miny),
                          float(bounds_max.maxx), float(bounds_max.maxy)]

        self.min_x, self.min_y = self.bound[0], self.bound[1]
        self.max_x, self.max_y = self.bound[2], self.bound[3]

    def getSize(self):
        example = self.grid_shp.loc[0]["geometry"]
        p_x, p_y = example.exterior.coords.xy
        size_x = p_x[1]-p_x[0]
        # remain the size of y as a negative to make consistent with raster data
        size_y = p_y[2]-p_y[1]
        return size_x, size_y

    def to_shapefile(self, loc):
        if loc[-3:] != "shp":
            loc = loc + ".shp"
        self.grid_shp.to_file(loc)


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

        # read information of shapefile inputted
        self.shp = gpd.read_file(in_shapefile)
        self.crs = self.shp.crs
        if self.crs != self.ras.GetProjection():
            raise Error("the crs doesn't match")

        self.ras = None

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

    def count_by_grid(self, typ, column=None, out_loc=None):
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
        if column:
            self.shp[column] = result.values()
        else:
            self.shp[typ] = result.values()
        # print(result)
        if out_loc:
            self.shp.to_file(out_loc)
        return Grid(self.shp)


class Guss:
    def __init__(self, popgrid, areagrid, threshold):
        self._pop = popgrid.grid_shp
        self._area = areagrid.grid_shp
        self.result = None
        self._threshold = threshold

    def getCenter(self, shp):
        p_x, p_y = shp.exterior.coords.xy
        return (p_x[1] + p_x[0]) / 2.0, (p_y[1] + p_y[2]) / 2.0

    def guss(self, dist):
        if dist > self._threshold:
            return 0
        q1 = math.exp(-(1 / 2.0) * (dist / self._threshold) * (dist / self._threshold))
        q2 = math.exp(-(1 / 2.0))
        return (q1 - q2) / (1 - q2)

    def close_selector(self, first, second, start, end):
        for _i in range(start, end):
            grid = first.loc[_i]
            __center = self.getCenter(grid["geometry"])
            center_point = shapely.Point(__center)
            buffer = center_point.buffer(self._threshold)
            _inter = second.intersects(buffer)
            _inter = second[_inter]
            # print(_i, _inter)
            yield _i, _inter

    def getDistance(self, inter_grids, _center):
        x_dis = self.getCenter(inter_grids)[0] - _center[0]
        y_dis = self.getCenter(inter_grids)[1] - _center[1]
        return math.sqrt(x_dis * x_dis + y_dis * y_dis)

    def run_step1(self, st, en, arr1):
        # step 1, green to people
        for i, inter in self.close_selector(self._area, self._pop, st, en):
            # print(i, inter)
            divisor = 0
            for inter_grid, inter_pop in zip(inter["geometry"], inter["pop"]):
                if inter_pop == 0:
                    continue
                center = self.getCenter(self._area.loc[i]["geometry"])
                distance = self.getDistance(inter_grid, center)
                divisor += self.guss(distance) * inter_pop
            if divisor == 0:
                rj = 0
            else:
                rj = self._area.loc[i]["area"] / divisor
                # print(rj)
            arr1[i] = rj
            # _area_grid.grid_shp.loc[i, "Rj"] = rj

    def run_step2(self, st, en, arr2):
        # step2 people to green
        for i, inter in self.close_selector(self._pop, self._area, st, en):
            ai = 0
            for inter_grid, inter_rj in zip(inter["geometry"], inter["Rj"]):
                if inter_rj == 0:
                    continue
                center = self.getCenter(self._pop.loc[i]["geometry"])
                distance = self.getDistance(inter_grid, center)
                ai += self.guss(distance) * inter_rj
            # _pop_grid.grid_shp.loc[i, "Ai"] = ai
            arr2[i] = ai

    def guss_reach(self, threads):
        if self._pop.crs != self._area.crs:
            raise Error("crs doesn't match")
        try:
            self._pop["pop"]
        except KeyError:
            raise Error("don't have a pop column, rename or add it to the pop grid")
        try:
            self._area["area"]
        except KeyError:
            raise Error("don't have a area column, rename or add it to the area grid")

        # self._area.loc[0, "Rj"], self._pop.loc[0, "Ai"] = 0, 0
        rja = multiprocessing.Array('f', size_or_initializer=self._pop.shape[0])
        aia = multiprocessing.Array('f', size_or_initializer=self._pop.shape[0])
        # print(start, step)
        # print(aia, rja)
        start, step = 0, int(self._pop.shape[0] / threads)
        for thread_id in range(threads):
            if thread_id != threads - 1:
                new_thread = multiprocessing.Process(target=self.run_step1, name=f"{thread_id}",
                                                     args=(start, start + step, rja))
            else:
                new_thread = multiprocessing.Process(target=self.run_step1, name=f"{thread_id}",
                                                     args=(start, self._pop.shape[0], rja))
            new_thread.start()
            new_thread.join()
            start += step

        self._area.loc[:, "Rj"] = [a for a in rja]
        self.result = self._area.copy()
        self.result.loc[:, "pop"] = self._pop["pop"]

        start, step = 0, int(self._pop.shape[0] / threads)
        for thread_id in range(threads):
            if thread_id != threads - 1:
                new_thread = multiprocessing.Process(target=self.run_step2, name=f"{thread_id}",
                                                     args=(start, start + step, aia))
            else:
                new_thread = multiprocessing.Process(target=self.run_step2, name=f"{thread_id}",
                                                     args=(start, self._pop.shape[0], aia))
            new_thread.start()
            new_thread.join()
            start += step

        self.result.loc[:, "Ai"] = [a for a in aia]

        return Grid(self.result)


class Error(Exception):
    def __init__(self, message):
        self.info = message


def raster_maker(out_loc, x, y, typ, tran, proj, new_data, background_value=-1):
    driver = gdal.GetDriverByName("GTiff")

    dts = driver.Create(
        out_loc,
        xsize=x,
        ysize=y,
        bands=1,
        eType=typ,
    )
    dts.SetGeoTransform(tran)
    dts.SetProjection(proj)
    dts.GetRasterBand(1).WriteArray(new_data)
    dts.GetRasterBand(1).SetNoDataValue(background_value)


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

    def get_conditions(inp, condition):
        """

        :param inp: a number or something else
        :param condition: a condition in form of ">n" etc.
        :return: True or False
        """
        if condition[0] == ">" and condition[1] != "=":
            return inp > float(condition[1:])
        elif condition[0] == "<" and condition[1] != "=":
            return inp < float(condition[1:])
        elif condition[0] == ">" and condition[1] == "=":
            return inp >= float(condition[2:])
        elif condition[0] == "<" and condition[1] == "=":
            return inp <= float(condition[2:])

    in_r = gdal.Open(file)
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

    raster_maker(save, x, y, typ, tran, proj, new_data, 0)


def make_grid(loc_in, step, loc_out=None):
    # ones the crs
    crs_file = open("crs.txt", "rb")
    crs = pickle.load(crs_file)
    crs_file.close()
    shp = gpd.read_file(loc_in).to_crs(crs)
    # make out the bound
    bounds = shp.bounds
    bounds_min = bounds.min()
    bounds_max = bounds.max()
    min_x = float(bounds_min.minx)
    max_x = float(bounds_max.maxx)
    min_y = float(bounds_min.miny)
    max_y = float(bounds_max.maxy)

    x, y = (min_x, max_y)

    # general
    # fixed point order left-up->right-up->right-down->left-down
    step = step
    geom_array = []
    while y >= min_y:
        while x <= max_x:
            geom = geometry.Polygon([(x, y), (x+step, y), (x+step, y-step), (x, y-step), (x, y)])
            geom_array.append(geom)
            x += step
        x = min_x
        y -= step

    # show
    grid = gpd.GeoDataFrame(geometry=geom_array, crs=crs)
    # the intersection of grid that intersect with the geometry
    """
    here was a part to be optimize
    """
    ind = grid.intersects(shp.geometry[0]).values
    for bound_shp in shp.geometry[1:]:
        ind = np.vstack([ind, grid.intersects(bound_shp).values])

    ind = ind.any(0)
    # cut
    grid = grid.loc[ind]
    grid = grid.reset_index().drop(columns="index")
    # fig, ax = plt.subplots(1, 2)
    # ax[0] = shp.plot(ax=ax[0])
    # ax[1] = grid.plot(ax=ax[1])
    # plt.show()
    if loc_out:
        grid.to_file(loc_out)
    return Grid(grid, size=(step, -step), bound=[min_x, min_y, max_x, max_y])


def normalize(data_line):
    _max = data_line.max()
    _min = data_line.min()
    # print(_max, _min)
    return (data_line - _min) / (_max-_min)


def demand(grid: Grid, ai_column_name, pop_column_name, out_loc):
    shp = grid.grid_shp
    for i in range(shp.shape[0]):
        try:
            ai = shp[ai_column_name][i]
            pop = shp[pop_column_name][i]
        except KeyError:
            raise Error("column name of Ai or pop can't be found")
        if pop == 0:
            ro = -1
        else:
            ro = ai/pop
        shp.loc[i, "RO"] = ro
    # data normalize
    shp.loc[shp["RO"] == -1, 'RO'] = shp["RO"].max()
    shp.loc[:, "RO"] = normalize(shp["RO"])
    grid.to_shapefile(out_loc)


def supply(grid: Grid, building_column_name, green_column_name, out_loc):
    shp = grid.grid_shp
    for i in range(shp.shape[0]):
        building_area = shp[i][building_column_name]
        green_area = shp[i][green_column_name]
        if building_area == 0:
            rp = -1
        else:
            rp = green_area/building_area
        shp.loc[i, "RP"] = rp

    grid.to_shapefile(out_loc)


def pop_raster_preprocess(pop_ras, out_loc):
    _pop_ras = gdal.Open(pop_ras)
    x, y = (_pop_ras.RasterXSize, _pop_ras.RasterYSize)
    typ = _pop_ras.GetRasterBand(1).DataType
    proj = _pop_ras.GetProjection()
    tran = _pop_ras.GetGeoTransform()

    new_data = _pop_ras.GetRasterBand(1).ReadAsArray().copy()
    for i in range(y):
        for j in range(x):
            if new_data[i, j] <= 0:
                new_data[i, j] = 0

    raster_maker(out_loc, x, y, typ, tran, proj, new_data, 0)


def raster_cutter(raster_in, mask: Grid, out_loc):
    file = gdal.Open(raster_in)
    proj = file.GetProjection()
    crs = mask.grid_shp.crs
    if crs != proj:
        option = gdal.WarpOptions(
            srcSRS=proj, dstSRS=crs,
            resampleAlg=gdal.GRA_Cubic, format="GTiff"
        )
        gdal.Warp('temp.tif', file, options=option)
        file = gdal.Open("temp.tif")
    tran = file.GetGeoTransform()
    typ = file.GetRasterBand(1).DataType
    # print(tran, "\n", mask.bound)
    _X_start, _Y_start = (math.floor((mask.bound[0] - tran[0])/tran[1]),
                          math.floor((mask.bound[3] - tran[3])/tran[5]))
    _X_end, _Y_end = (math.ceil((mask.bound[2] - tran[0])/tran[1]),
                      math.ceil((mask.bound[1] - tran[3])/tran[5]))
    new_data = file.GetRasterBand(1).ReadAsArray()[_Y_start:_Y_end, _X_start:_X_end]
    # print(new_data)
    x, y = (_X_end - _X_start, _Y_end - _Y_start)
    new_tran = (mask.bound[0], tran[1], tran[2], mask.bound[3], tran[4], tran[5])
    raster_maker(out_loc, x, y, typ, new_tran, proj, new_data)
    # close the file
    file = None
    if os.path.exists("temp.tif"):
        os.remove("temp.tif")


if __name__ == "__main__":
    '''
    # extract_by_value("Raster/using/2000t.tif", "test.tif", 41, 42, 43, 44, 45, 46)
    g = make_grid("ShapeFile/T.shp", 1000, "Raster_out/grid.shp")
    zonal_er = Zonal("test.tif", "Raster_out/grid.shp")
    area_grid = zonal_er.count_by_grid("area")
    pop_raster_preprocess("Raster/a2000/hdr.adf", "Raster_out/pop.tif")
    zonal_er = Zonal("Raster_out/pop.tif", "Raster_out/grid.shp")
    pop_grid = zonal_er.count_by_grid("sum")
    pop_grid.grid_shp["pop"] = pop_grid.grid_shp["sum"]
    pop_grid.grid_shp = pop_grid.grid_shp.drop(columns="sum")
    print(pop_grid.bound, pop_grid.size)
    guss_cal = Guss(pop_grid, area_grid, 2500)
    test = guss_cal.guss_reach(15)
    demand(test, "Ai", "pop", "test")
    # print(test)
    '''
    g = make_grid("ShapeFile/T.shp", 1000, "Raster_out/grid.shp")
    raster_cutter("dataverse_files/DMSP1992.tif", g, "new.tif")
