import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import tqdm
import tqdm.notebook
#import librosa
#import librosa.display
import matplotlib.dates as mdates
import numpy as np
import glob
import h5py
import os
import re
#import geopandas
import utils


DESCRIPTIVE_COLS = ('near_commercial', 'near_construction', 'near_dogpark', 'near_highway', 'near_park',
                    'near_touristspot', 'near_transporthub', 'nyu_location', 'nyu_surroundings', 'on_thoroughfare')

SPL_AGG_COLUMNS = ('l1', 'l10', 'l5', 'l90', 'laeq', 'kurtosis', 'std', 'mean', 'skew', 'median', 'max', 'min')

CLASSES_FINE = ['small-sounding-engine', 'medium-sounding-engine', 'large-sounding-engine',
          'rock-drill', 'jackhammer', 'hoe-ram', 'pile-driver',
          'non-machinery-impact','chainsaw', 'small-medium-rotating-saw', 'large-rotating-saw',
          'car-horn', 'car-alarm', 'siren', 'reverse-beeper',
          'stationary-music', 'mobile-music', 'ice-cream-truck',
          'person-or-small-group-talking', 'person-or-small-group-shouting',
          'large-crowd', 'amplified-speech','dog-barking-whining']

CLASSES_COARSE = ['Engine', 'Machinery Impact', 'Non-Machinery Impact',
              'Powered Saw', 'Alert Signal', 'Music', 'Human Voice', 'Dog']

def limit_to_dates(df, date_vals, min_date_str, max_date_str):
    """
    Limit to dates for covid analysis

    Parameters
    ----------
    df
    max_date_str
    min_date_str

    Returns
    -------
    df
    """
    df = df[((date_vals >= datetime.datetime.strptime('2017-{}'.format(min_date_str), '%Y-%m-%d').date()) &
             (date_vals <= datetime.datetime.strptime('2017-{}'.format(max_date_str), '%Y-%m-%d').date())) |
            ((date_vals >= datetime.datetime.strptime('2018-{}'.format(min_date_str), '%Y-%m-%d').date()) &
             (date_vals <= datetime.datetime.strptime('2018-{}'.format(max_date_str), '%Y-%m-%d').date())) |
            ((date_vals >= datetime.datetime.strptime('2019-{}'.format(min_date_str), '%Y-%m-%d').date()) &
             (date_vals <= datetime.datetime.strptime('2019-{}'.format(max_date_str), '%Y-%m-%d').date())) |
            ((date_vals >= datetime.datetime.strptime('2020-{}'.format(min_date_str), '%Y-%m-%d').date()) &
             (date_vals <= datetime.datetime.strptime('2020-{}'.format(max_date_str), '%Y-%m-%d').date()))]
    return df


def load_spl_data(spl_hdf5_path, do_limit_to_covid_dates=False,
                  add_in_endpoints=False, min_date_str=None, max_date_str=None,
                  exclude_sensors=('sonycnode-b827ebefb215.sonyc', 'sonycnode-b827eb122f0f.sonyc',
                                   'sonycnode-b827eb905497.sonyc', 'sonycnode-b827eb0d8af7.sonyc')):
    """
    SPL data loader

    Parameters
    ----------
    spl_hdf5_path
    do_limit_to_covid_dates
    add_in_endpoints
    min_date_str
    max_date_str
    exclude_sensors

    Returns
    -------
    df
    """
    df = pd.read_hdf(spl_hdf5_path)

    if add_in_endpoints and min_date_str is not None and max_date_str is not None:
        df = add_in_end_points(df, min_date_str, max_date_str)

    # fix types
    df['address'] = df['address'].str.decode('utf-8')
    #df['sensor_id'] = df['sensor_id'].str.decode('utf-8')
    df['bus_route'] = df['bus_route'].astype(bool)
    df['height_ft'] = df['height_ft'].astype(int)
    df['hour_of_day'] = df['hour_of_day'].astype(int)
    df['roadway_width_ft'] = df['roadway_width_ft'].astype(int)
    for c in DESCRIPTIVE_COLS:
        df[c] = df[c].astype(bool)

    # weekday vs weekend
    df.loc[df['weekday'] < 5, 'period'] = 'weekday'
    df.loc[df['weekday'] >= 5, 'period'] = 'weekend'

    # fix dates to be backwards compatible with older code
    df['datetime'] = df['date']
    df['date'] = df['date'].dt.date
    df['year'] = df['datetime'].dt.year

    # add in year groups
    df.loc[(df['year'] < 2020) & (df['year'] >= 2017), 'year_group'] = '2017-2019'
    df.loc[df['year'] == 2020, 'year_group'] = '2020'

    # add in descriptive title to column
    for sensor_id in df['sensor_id'].unique():
        df.loc[df['sensor_id'] == sensor_id, 'sensor_title'] = get_descriptive_title(sensor_id, df)

    if do_limit_to_covid_dates:
        df = limit_to_dates(df, df['date'], min_date_str, max_date_str)

    if exclude_sensors is not None:
        df = df[~df['sensor_id'].isin(exclude_sensors)]

    # melt data
    id_cols = list(set(df.columns.to_list()) - set(SPL_AGG_COLUMNS))
    df = pd.melt(df, id_vars=id_cols, value_vars=SPL_AGG_COLUMNS)

    return df


def add_in_end_points(df, min_date_str, max_date_str, descriptive_cols=DESCRIPTIVE_COLS):
    """
    Add in null valued data at `min_date_str` and `max_date_str` so that we can use resampling between these dates.

    Parameters
    ----------
    df : pandas.DataFrame
    min_date_str : str
    max_date_str : str
    descriptive_cols : list(str)
        The columns describing sensor attributes


    Returns
    -------
    df : pandas.DataFrame
    """
    descriptive_cols = list(descriptive_cols) + ['lat', 'lng', 'height_ft', 'roadway_width_ft']
    df = df.copy()
    dates = ['2017-{}'.format(min_date_str), '2017-{}'.format(max_date_str),
             '2018-{}'.format(min_date_str), '2018-{}'.format(max_date_str),
             '2019-{}'.format(min_date_str), '2019-{}'.format(max_date_str),
             '2020-{}'.format(min_date_str), '2020-{}'.format(max_date_str)]
    for sensor_id in tqdm.tqdm_notebook(df['sensor_id'].unique()):
        d_col_vals = df.loc[df['sensor_id'] == sensor_id, descriptive_cols].drop_duplicates().values.reshape(-1)
        for date in dates:
            df.loc[-1] = np.nan
            df.loc[-1, 'date'] = datetime.datetime.strptime(date, '%Y-%m-%d')
            df.loc[-1, 'weekday'] = df.loc[-1, 'date'].weekday()
            df.loc[-1, 'sensor_id'] = sensor_id
            df.loc[-1, 'hour_of_day'] = 0
            df.loc[-1, descriptive_cols] = d_col_vals
            df.index = df.index + 1
            df = df.sort_index()

    df['hour_of_day'] = df['hour_of_day'].astype('int')
    df['weekday'] = df['weekday'].astype('int')
    return df


def calculate_change_in_value(df, var_column='variable'):
    """
    Calculate the change in value in comparison to historical data

    Parameters
    ----------
    df : pandas.DataFrame
    var_column : str
        The name of the variable column

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()
    
    past_values = df[(df['year']!=2020)].groupby(['sensor_id','hour_of_day', 'weekday', var_column])['value'].mean().to_frame('past_value').reset_index()
    past_values['year'] = 2020
    df = df.merge(past_values, how='left')
    df['value_change'] = df['value'] - df['past_value']

    return df


def add_aligned_hour_index(df, min_year=2017, max_year=2020):
    df = df.copy()

    max_weekday = max([df.loc[df['year'] == year, 'date'].min().weekday() for year in range(min_year, max_year+1)])
    for year in range(min_year, max_year+1):
        min_date = df.loc[df['year'] == year, 'date'].min()
        offset = max_weekday - datetime.datetime.fromordinal(min_date.toordinal()).weekday()
        baseday = min_date + datetime.timedelta(days=offset)
        df.loc[df['year'] == year, 'aligned_day_index'] = (df.loc[df['year'] == year, 'date'] - baseday).dt.days

    df['aligned_hour_index'] = df['aligned_day_index'] * 24 + df['hour_of_day']
    df = df[df['aligned_hour_index'] >= 0]
    return df

def add_aligned_day_index(df, min_year=2017, max_year=2020):
    df = df.copy()

    max_weekday = max([df.loc[df['year'] == year, 'date'].min().weekday() for year in range(min_year, max_year+1)])
    for year in range(min_year, max_year+1):
        min_date = df.loc[df['year'] == year, 'date'].min()
        offset = max_weekday - datetime.datetime.fromordinal(min_date.toordinal()).weekday()
        baseday = min_date + datetime.timedelta(days=offset)
        df.loc[df['year'] == year, 'aligned_day_index'] = (df.loc[df['year'] == year, 'date'] - baseday).dt.days

    df = df[df['aligned_day_index'] >= 0]
    return df


def calculate_aligned_change_in_value(df, var_column='variable', min_year=2017, max_year=2020):
    """
    Using the offset alignment in `add_aligned_hour_index` average over years to calculate change. This is going to be noisier

    Parameters
    ----------
    df
    var_column

    Returns
    -------

    """
    df = df.copy()
    df = add_aligned_hour_index(df, min_year, max_year)

    past_values = df[(df['year'] != 2020)].groupby(['sensor_id', 'aligned_hour_index', var_column])[
        'value'].mean().to_frame('aligned_past_value').reset_index()
    past_values['year'] = 2020
    df = df.merge(past_values, how='left')
    df['aligned_value_change'] = df['value'] - df['aligned_past_value']

    return df


def calculate_rolling_change_in_value(df, rolling_interval=15, var_column='variable', value_column='value',
                                      on='aligned_day_index', center=True, win_type='boxcar'):
    """
    Calculate the change in value in comparison to historical data by calculating grouping by weekday and hour, but then
    calculating a rolling average over the `rolling_interval`. When aggregating years, align by week.

    Parameters
    ----------
    df : pandas.DataFrame
    rolling_interval : str or int
        datetime interval to calculate average over
    var_column : str
        The name of the variable column
    value : str
        Te name of the value column
    on : str
        The column to roll on
    center : bool
        Center the rolling window
    win_type : str
        The type of window for rolling average. See pandas rolling docs.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    # use week for alignment
    df['week'] = df['datetime'].dt.week

    interval_smoothed_values = df.groupby(['sensor_id', 'hour_of_day', 'weekday', var_column]).rolling(rolling_interval, on=on,
                                                                                                       center=center, win_type=win_type,
                                                                                                       min_periods=1)[
        value_column].mean().to_frame('interval_smoothed_value').reset_index()
    _df = df.merge(interval_smoothed_values, how='left')
    rolling_past_values = _df[(_df['year'] != 2020)].groupby(['sensor_id', 'week', 'weekday', 'hour_of_day', var_column])[
        'interval_smoothed_value'].mean().to_frame('rolling_past_value').reset_index()
    rolling_past_values['year'] = 2020
    df = df.merge(rolling_past_values, how='left')
    df['rolling_value_change'] = df[value_column] - df['rolling_past_value']

    return df


def get_descriptive_title(sensor_id, df, descriptive_cols=DESCRIPTIVE_COLS):
    """
    Get title for plot with descriptive attributes

    Parameters
    ----------
    sensor_id : str
    df : pandas.DataFrame
    descriptive_cols : list(str)
        The columns describing sensor attributes

    Returns
    -------
    title : str
    """
    descriptive_cols = list(descriptive_cols) + ['lat', 'lng', 'height_ft']
    row = df.loc[df['sensor_id'] == sensor_id, descriptive_cols].drop_duplicates()
    title = sensor_id
    title += ' ({:04.2f}, {:04.2f})'.format(float(row['lat']),float(row['lng']))
    for c in descriptive_cols:
        if (row[c] == 1).any():
            title += ' / {}'.format(c)
    title += ' / {}ft'.format(row['height_ft'].values[0])
    return title


def load_hires_spl_df(hires_df_path, exclude_sensors, min_data_points=12000, min_date=None, max_date=None):
    df_orig = pd.read_hdf(hires_df_path)
    df_orig.index = df_orig.index.tz_localize(None)
    df_orig.index = df_orig.index.rename('datetime')
    df_orig['date'] = df_orig.index.date
    df_orig['datetime'] = df_orig.index
    df_orig['offset'] = df_orig['datetime'] - (pd.to_datetime(df_orig['date']) + datetime.timedelta(hours=18, minutes=45))
    df_orig = df_orig[~df_orig['sensor_id'].isin(exclude_sensors)]

    # remove days without a certain number of values in the time frame
    rows = []
    for sensor_id in df_orig['sensor_id'].unique():
        _df = df_orig[(df_orig['sensor_id'] == sensor_id) & (df_orig['date'])]
        for dt in df_orig[df_orig['sensor_id'] == sensor_id]['date'].unique():
            __df = _df[(_df['date'] == dt)]
            if __df['offset'].count() > min_data_points:
                rows.append(__df)
    df_orig = pd.concat(rows)
    df_orig.index = df_orig['datetime']

    if min_date is not None:
        df_orig = df_orig[df_orig.index.date >= min_date]

    if min_date is not None:
        df_orig = df_orig[df_orig.index.date <= max_date]

    del df_orig['datetime']
    return df_orig


def calc_clapping_stats(df, start_offset, stop_offset, df_l90):
    stats = []
    for date in tqdm.tqdm_notebook(np.sort(df['date'].unique())):
        for sensor_id in df['sensor_id'].unique():
            _df = df[(df['date'] == date) & (df['sensor_id'] == sensor_id)]
            try:
                l90 = df_l90.loc[(df_l90['date'] == date) & (df_l90['sensor_id'] == sensor_id), 'l90'].values[0]
                peak = _df.loc[(_df['offset'] >= pd.Timedelta(minutes=start_offset)) &
                               (_df['offset'] <= pd.Timedelta(minutes=stop_offset)), 'dBAS'].values.max()
                stats.append(dict(sensor_id=sensor_id, date=date, l90=l90, peak=peak))
            except Exception as e:
                print(sensor_id, date, e)
                continue

    df_stats = pd.DataFrame.from_records(stats)
    df_stats['diff'] = df_stats['peak'] - df_stats['l90']
    return df_stats


def get_clapping_intensity_dfs(hires_df_path, node_info_df, output_dir, start_offset=15, stop_offset=18,
                               resample_interval='15s',
                               exclude_sensors=('sonycnode-b827ebefb215.sonyc',
                                                'sonycnode-b827eb122f0f.sonyc',
                                                'sonycnode-b827eb905497.sonyc',
                                                'sonycnode-b827eb0d8af7.sonyc',
                                                'sonycnode-b827eb491436.sonyc',
                                                'sonycnode-b827ebe3b72c.sonyc',
                                                'sonycnode-b827ebf31214.sonyc'),
                               min_date=None, max_date=None):

    df_orig = load_hires_spl_df(hires_df_path, exclude_sensors, min_date=min_date, max_date=max_date)

    df_l90 = df_orig.groupby(['date', 'sensor_id'])['dBAS'].apply(utils.calcl90).to_frame('l90').reset_index()
    df_l90['date'] = df_l90['date'].dt.date

    # calc laeq over resample_interval
    df = df_orig.groupby([pd.Grouper(freq='D'), 'sensor_id']).resample(resample_interval)['dBAS'].mean().apply(
        utils.calc_leq)
    df = df.reset_index(level=0)
    df['date'] = df['datetime'].dt.date
    del df['datetime']
    df = df.reset_index()

    df['offset'] = df['datetime'] - (pd.to_datetime(df['date']) + datetime.timedelta(hours=18, minutes=45))

    df_stats = calc_clapping_stats(df, start_offset, stop_offset, df_l90)

    df_stats = df_stats.merge(node_info_df[['location_id', 'sensor_id']])
    del df_stats['sensor_id']

    df_stats = pd.melt(df_stats, id_vars=['location_id', 'date'])
    df_stats = df_stats.sort_values(['location_id', 'date', 'variable'])
    df_stats.to_csv(os.path.join(output_dir, 'clapping_intensity_above_ambient.csv'), index=False)

    df = df.merge(node_info_df[['location_id', 'sensor_id']])
    df = df[['datetime', 'date', 'offset', 'location_id', 'dBAS']]
    df.columns = ['datetime', 'date', 'offset', 'location_id', 'laeq']
    df = df.sort_values(['location_id', 'datetime'])
    df.to_csv(os.path.join(output_dir, 'laeq_clapping_intervals.csv'), index=False)

    return df, df_stats


# DATA EXPORT
def get_node_info(df, mappluto_path, descriptive_cols=DESCRIPTIVE_COLS, output_path=None):
    """
    Get node info w/ quantized location info used pluto data

    Parameters
    ----------
    df : pandas.DataFrame
    mappluto_path : str
    descriptive_cols : list(str)
        The columns describing sensor attributes
    output_path : str

    Returns
    -------
    sensor_gdf : geopandas.DataFrame
    """
    cols = list(descriptive_cols) + ['sensor_id', 'lat', 'lng']
    node_info_df = df[cols].drop_duplicates().sort_values('sensor_id').reset_index(drop=True)
    node_info_df['node_id'] = node_info_df.index

    if mappluto_path is not None:
        # Create sensor dataframe and merge with pluto data
        sensor_gdf = geopandas.GeoDataFrame(node_info_df,
                                            geometry=geopandas.points_from_xy(node_info_df['lng'],
                                                                              node_info_df['lat']),
                                            crs={'init': 'epsg:4326'})
        sensor_gdf = sensor_gdf.to_crs({'init': 'epsg:3857'})

        # Since some of the sensors locations are not within lots (are on the street), add a radius and the find what intersects
        sensor_gdf['geometry'] = sensor_gdf.buffer(15)
        sensor_gdf = sensor_gdf.to_crs({'init': 'epsg:4326'})

        pluto = geopandas.read_file(mappluto_path)
        pluto = pluto.to_crs({'init': 'epsg:4326'})

        # Get pluto data for sensor locations
        sensor_gdf = geopandas.sjoin(sensor_gdf, pluto, how='left', op='intersects')

        sensor_gdf = sensor_gdf.drop_duplicates(subset=['sensor_id'], keep='first')[
            ['sensor_id', 'Borough', 'Block', 'Latitude', 'Longitude'] + list(descriptive_cols)]

        # Use codes from BBL
        sensor_gdf.loc[sensor_gdf['Borough'] == 'MN', 'Borough'] = 1
        sensor_gdf.loc[sensor_gdf['Borough'] == 'BK', 'Borough'] = 3
        sensor_gdf.loc[sensor_gdf['Borough'] == 'QN', 'Borough'] = 4

        sensor_gdf = sensor_gdf[list(descriptive_cols) + ['sensor_id', 'Borough', 'Block', 'Latitude', 'Longitude']]
        sensor_gdf.columns = list(descriptive_cols) + ['sensor_id', 'borough', 'block', 'bb_lat', 'bb_lng']

        output_df = sensor_gdf
    else:
        output_df = node_info_df

    if output_path is not None:
        output_df.to_csv(output_path)

    return output_df


def export_day_rank_csv(spl_hdf5_path, output_path, start_idx, stop_idx, min_sensors=14):
    df_full = load_spl_data(spl_hdf5_path)
    day_rank_df = df_full[df_full['variable'] == 'laeq'].groupby(['date', 'sensor_id'])['value'].apply(utils.calc_leq).groupby(
        'date').describe().sort_values('mean').reset_index()[['date', 'mean', 'count']]
    print('{} coverage'.format(np.sum(day_rank_df['count'] >= min_sensors) / day_rank_df['count'].shape[0]))
    day_rank_df = day_rank_df[day_rank_df['count'] >= min_sensors]
    day_rank_df = day_rank_df[start_idx:stop_idx][['date', 'mean']]
    day_rank_df.columns = ['date', 'avg_laeq']
    day_rank_df.to_csv(output_path, index=False)
    return day_rank_df


def export_spl_daily_csv(df, node_info_df, output_path, drop_null_aligned_date=True):
    _df = df[df['variable'] == 'laeq'].copy()
    _df = _df.groupby(['sensor_id', 'date', 'aligned_day_index'])['value'].apply(utils.calc_leq).reset_index()
    _df = _df.merge(node_info_df, on='sensor_id')
    _df['datetime'] = pd.to_datetime(_df['date'])
    _df['year'] = _df['datetime'].dt.year
    _df = _df.merge(_df[_df['year'] == 2020][['aligned_day_index', 'date']].drop_duplicates(), on='aligned_day_index', how='left')
    _df = _df[['location_id', 'date_x', 'date_y', 'year', 'value']]
    _df.columns = ['location_id', 'date', 'aligned_2020_date', 'year', 'laeq']
    if drop_null_aligned_date:
        _df = _df[~pd.isna(_df['aligned_2020_date'])]
    _df = _df.sort_values(['location_id', 'date'])
    _df.to_csv(output_path, index=False)


def export_spl_hourly_change_csv(df, node_info_df, output_path):
    _df = df[(df['year'] == 2020) & (df['variable'] == 'laeq')].copy()
    _df = _df.merge(node_info_df, on='sensor_id')[['location_id', 'datetime', 'value',
                                                   'rolling_past_value', 'rolling_value_change']]
    _df.columns = ['location_id', 'datetime', 'laeq', 'past_laeq', 'laeq_change']
    _df = pd.melt(_df, id_vars=['location_id', 'datetime'])
    _df.columns = ['location_id', 'datetime', 'variable', 'value']
    _df = _df.sort_values(['location_id', 'datetime', 'variable'])
    _df.to_csv(output_path, index=False)


def export_spl_daily_change_csv(df, node_info_df, output_path):
    _df = df[(df['year'] == 2020) & (df['variable'] == 'laeq')].copy()
    _df = _df.groupby(['sensor_id', 'date']).mean().reset_index()
    _df = _df.merge(node_info_df, on='sensor_id')[['location_id', 'date', 'value',
                                                   'rolling_past_value', 'rolling_value_change']]
    _df.columns = ['location_id', 'date', 'laeq', 'past_laeq', 'laeq_change']
    _df = pd.melt(_df, id_vars=['location_id', 'date'])
    _df.columns = ['location_id', 'date', 'variable', 'value']
    _df = _df.sort_values(['location_id', 'date', 'variable'])
    _df.to_csv(output_path, index=False)


def process_sound_class_presence_for_export(csv_input_path, csv_output_path, node_info_df):
    df = pd.read_csv(csv_input_path)
    print('{} input sound class presence rows'.format(df.shape[0]))
    df['sensor_id'] = 'sonycnode-' + df['sensor_id'] + '.sonyc'
    df = df.merge(node_info_df[['location_id', 'sensor_id']])

    df['datetime'] = pd.to_datetime(df['date'])
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['hour_of_day'] = 0
    df = add_aligned_hour_index(df)
    df = df.merge(df[df['year'] == 2020][['aligned_day_index', 'date']].drop_duplicates(), on='aligned_day_index',
                  how='left')
    df = df[['location_id', 'date_x', 'date_y', 'sound_class', 'presence']]
    df.columns = ['location_id', 'date', 'aligned_2020_date', 'sound_class', 'presence']
    df = df[~pd.isna(df['aligned_2020_date'])]
    df = df.sort_values(['location_id', 'date', 'sound_class'])
    print('{} output sound class presence rows'.format(df.shape[0]))
    df.to_csv(csv_output_path, index=False)


def process_sound_class_presence_change_for_export(csv_input_path, csv_output_path, node_info_df):
    df = pd.read_csv(csv_input_path)
    df['sensor_id'] = 'sonycnode-' + df['sensor_id'] + '.sonyc'
    df = df.merge(node_info_df[['location_id', 'sensor_id']])
    df = df[['location_id', 'date', 'sound_class', 'variable', 'value']]
    df = df.sort_values(['location_id', 'date', 'sound_class'])
    df.to_csv(csv_output_path, index=False)


def nyt_export(spl_hdf5_path, output_dir, mappluto_path, min_date_str, max_date_str, remove_sensor_id=True, min_sensors=15):
    """
    Export for NYT piece

    Parameters
    ----------
    spl_hdf5_path
    output_dir
    min_date_str
    max_date_str

    Returns
    -------

    """
    export_day_rank_csv(spl_hdf5_path, os.path.join(output_dir, 'day_rank_avg_laeq.csv'), 0, 30, min_sensors=min_sensors)

    # get just laeq
    df = load_spl_data(spl_hdf5_path)

    df = add_aligned_hour_index(df)
    df = calculate_rolling_change_in_value(df, rolling_interval=15, on='aligned_day_index', center=True)

    # get node info
    node_info_df = get_node_info(df, mappluto_path)
    node_info_df.loc[-1, 'sensor_id'] = 'sonycnode-b827eb491436.sonyc'
    node_info_df.loc[-2, 'sensor_id'] = 'sonycnode-b827ebe3b72c.sonyc'
    node_info_df.loc[-3, 'sensor_id'] = 'sonycnode-b827eb905497.sonyc'
    node_info_df.loc[-4, 'sensor_id'] = 'sonycnode-b827eb0d8af7.sonyc'

    node_info_df = node_info_df.sort_values('sensor_id').reset_index(drop=True)
    node_info_df['location_id'] = node_info_df.index + 1
    node_info_df = node_info_df[np.roll(node_info_df.columns, 1)]

    export_spl_daily_csv(df[df['date'] >= datetime.date(2019, 1, 1)], node_info_df,
                         os.path.join(output_dir, 'laeq_daily_2019-2020_allmonths.csv'),
                         drop_null_aligned_date=False)

    df = limit_to_dates(df, df['date'], min_date_str, max_date_str)
    export_spl_daily_csv(df, node_info_df, os.path.join(output_dir, 'laeq_daily_2017-2020.csv'))
    export_spl_hourly_change_csv(df, node_info_df, os.path.join(output_dir, 'laeq_hourly_change.csv'))
    export_spl_daily_change_csv(df, node_info_df, os.path.join(output_dir, 'laeq_daily_change.csv'))

    process_sound_class_presence_for_export('../exports/sound_classes_presence_2017-2020_raw.csv',
                                            '../exports/sound_classes_presence_2017-2020.csv',
                                            node_info_df)

    process_sound_class_presence_change_for_export('../exports/sound_classes_presence_change_2020_raw.csv',
                                                   '../exports/sound_classes_presence_change_2020.csv',
                                                   node_info_df)

    if remove_sensor_id:
        del node_info_df['sensor_id']
    node_info_df.to_csv(os.path.join(output_dir, 'locations.csv'), index=False)

    return df, node_info_df


def nyt_audio_export(audio_h5_path, output_path, SPL_DATAPATH='../data/spl/spl.h5'):
    # fine class data
    print('loading fine class data..')
    df_fine = load_audio_data(audio_h5_path,
                              CLASSES_FINE,
                              classes_of_interest=['person-or-small-group-talking', 'car-horn', 'siren'],
                              class_granularity='fine', SPL_DATAPATH=SPL_DATAPATH)
    df_fine = compute_presence(df_fine, thresholds={'car-horn': 0.3})
    df_fine = add_aligned_hour_index(df_fine)
    df_fine = calculate_rolling_change_in_value(df_fine, var_column='sound_class', value_column='presence',
                                                rolling_interval=15, on='aligned_day_index', center=True)

    # coarse class data
    print('loading coarse class data..')
    df_coarse = load_audio_data(audio_h5_path,
                                CLASSES_COARSE,
                                classes_of_interest=['Engine'],
                                class_granularity='coarse', SPL_DATAPATH=SPL_DATAPATH)
    df_coarse = df_coarse.replace('Engine', 'engine')
    df_coarse = compute_presence(df_coarse, {'engine': 0.9})
    df_coarse = add_aligned_hour_index(df_coarse)
    df_coarse = calculate_rolling_change_in_value(df_coarse, var_column='sound_class', value_column='presence',
                                                  rolling_interval=15, on='aligned_day_index', center=True)

    df_all = pd.concat((df_fine, df_coarse))

    print('exporting csv files..')
    df_presence = export_daily_presence(df_all, os.path.join(output_path, 'sound_classes_presence_2017-2020.csv'))
    df_change = export_daily_change_presence(df_all,
                                             os.path.join(output_path, 'sound_classes_presence_change_2020.csv'))

    return df_presence, df_change


def export_daily_presence(df, output_path):
    df_presence = df.groupby(['sensor_id', 'date', 'sound_class'])['presence'].mean().reset_index()
    df_presence.to_csv(output_path, index=False)
    return df_presence


def export_daily_change_presence(df, output_path):
    df_change = df[df['year'] == 2020]
    df_change = df_change.groupby(['sensor_id', 'date', 'sound_class'])['rolling_value_change',
                                                                        'rolling_past_value',
                                                                        'presence'].mean().reset_index()
    df_change['presence_change'] = df_change['rolling_value_change']
    df_change['past_presence'] = df_change['rolling_past_value']
    df_change = df_change.drop(['rolling_value_change'], axis=1)
    df_change = df_change.drop(['rolling_past_value'], axis=1)
    df_change = pd.melt(df_change,
                        id_vars=['sensor_id', 'date', 'sound_class'],
                        value_vars=['presence', 'presence_change', 'past_presence'])
    df_change.to_csv(output_path, index=False)
    return df_change


# PLOTTING
def plot_matrix_w_dates(X, x_coords, y_coords):
    """
    Matrix plot with dates on either x or y. Assign `None` to x_coords or y_coords if you don't want dates.

    Parameters
    ----------
    X
    x_coords
    y_coords
    dim

    Returns
    -------

    """
    ax = librosa.display.specshow(X,
                                  x_coords=x_coords, y_coords=y_coords)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    if x_coords is not None:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    if y_coords is not None:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
    return ax


def add_in_weekend_shading(ax, data, granularity='datetime'):
    '''
    Add shading in weekends in plot.

    Parameters
    ----------
    ax
    data
    granularity

    '''
    if ('datetime' not in data.columns) and ('date' in data.columns):
        granularity = 'date'
    data = data.sort_values(granularity)
    data = data.drop_duplicates([granularity, 'weekday'])
    amin, amax = ax.get_ylim()
    ax.fill_between(data[granularity], amin, amax, where=data['weekday'] >= 5,
                facecolor='grey', alpha=0.1)


def lineplot(df, x, y, new_plot=True, ax=None, title=None, label=None, ylabel='Change',
             important_dates=None, baseline=False, color='k', hue=None, palette=None):
    '''
    Creates seaborn lineplots to ilustrate the change of a signal (e.g. SPL or class presence),
    adding a baseline, important dates and weekends for covid-related analysis.


    Parameters
    ----------
    df (pandas.DataFrame):
        Dataframe with data to plot
    x (str or data):
        Name of DataFrame columns to plot in x-axis. It can also be data directly.
    y (str or data):
        Name of DataFrame columns to plot in x-axis. It can also be data directly.
    new_plot (bool):
        If True creates a new matplotlib figure. Set False if want to plot different data on same figure.
    title (str or None):
        Title of plot
    label (str or None):
        Label of data
    ylabel (str or None):
        Label for y axis. Default is 'Change'.
    important_dates (dict or None):
        Dictionary with format {str: datetime.date} containing relevant dates regarding the covid analysis.
    baseline (bool):
        If True it includes the baseline indicating 0 change
    color (str or None):
        Color to use for ploting

    Returns
    -------
    ax (matplotlib ax):
        axis with given plot

    '''

    if not df.isnull().values.all():

        if new_plot:
            fig, ax = plt.subplots(figsize=(20, 3))
            plt.title(title, fontsize=16)
            plt.xticks(rotation=30, ha='right')  # here?

        sns.lineplot(data=df, x=x, y=y, label=label, color=color, palette=palette, hue=hue, ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time')

        if baseline:
            ax.axhline(0, c='red', ls='--', alpha=0.7)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.text(xmin, ymax*0.1, 'Baseline', color='red')

        if important_dates is not None:
            ymin, ymax = ax.get_ylim()
            # deltay = np.arange(0, len(important_dates), 0.05)
            for deltay, data in enumerate(important_dates.items()):
                event = data[0]
                date = data[1]
                ax.axvline(date, ymin-1, ymax+1, color='k', alpha=0.3, linestyle='--')
                ax.text(date, ymax-0.1*(ymax-ymin)*deltay, event)

        if ('weekday' in df.columns):
            add_in_weekend_shading(ax, df)

        return ax


# AUDIO  ======================================================================

def load_audio_data(research_workspace, classes, classes_of_interest=None, class_granularity='fine',
                    exclude_sensors=('b827ebefb215', 'b827eb122f0f', 'b827eb491436', 'b827ebe3b72c',
                                     'b827eb905497', 'b827eb0d8af7'),
                    do_limit_to_covid_dates=False, min_date_str=None, max_date_str=None,
                    SPL_DATAPATH='../data/spl/spl.h5'):
    '''

    Function to read h5 files with the model predictions and format it into a DataFrame for analysis.
    This function takes time to finish processing. Think of saving the result.

    Parameters
    ----------
    research_workspace (str):
        Folder to h5 files.
    classes (list of str):
        List of classes to name the columns of the DataFrame.
    classes_of_interest (list or None):
        List of classes of interest, used to restrict and speed up loading
    class_granularity (str):
        Either fine or coarse.
    exclude_sensors (list of str):
        list of sensors to exclude from analysis
    do_limit_to_covid_dates (bool):
        Wheter or not limit the analysis dates to a specific period
    min_date_str (str):
        Month and day of the lower day limit of the analysis period
    max_date_str (str):
    Month and day of the upper day limit of the analysis period
        SPL_DATAPATH (str):
        Path to spl.h5 file with SPL analysis data


    Returns
    -------
    df (pandas DataFrame):
        DataFrame with datetime, sensor and predictions information.

    '''

    pat = '(.+)_(\d+\.\d\d)_ol3.npz'
    prog = re.compile(pat)

    dfs = []
    for f in glob.glob(os.path.join(research_workspace, '*.h5')):
        h5 = h5py.File(f, 'r')
        d = h5[class_granularity]
        sensor_ids = [prog.match(f.decode('utf8')).group(1) for f in d['filename']]
        timestamps = [float(prog.match(f.decode('utf8')).group(2)) for f in d['filename']]

        _df = pd.DataFrame.from_dict(dict([(c, d[c]) for c in d.dtype.names]))
        del _df['timestamp']
        _df['sensor_id'] = sensor_ids
        _df['timestamp'] = timestamps
        _df['filename'] = _df['filename'].str.decode('utf8')
        _df['datetime'] = pd.DatetimeIndex(pd.to_datetime(_df['timestamp'].astype('int'),
                                                          unit='s', utc=True)).tz_convert('America/New_York')
        _df['weekday'] = _df['datetime'].dt.weekday
        _df['hour_of_day'] = _df['datetime'].dt.hour
        _df['year'] = _df['datetime'].dt.year
        _df['date'] = _df['datetime'].dt.date
        _df['sensor_id'] = sensor_ids

        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = ['filename'] + classes + ['sensor_id', 'timestamp', 'datetime', 'weekday', 'hour_of_day', 'year',
                                           'date']
    # weekday vs weekend
    df.loc[df['weekday'] < 5, 'period'] = 'weekday'
    df.loc[df['weekday'] >= 5, 'period'] = 'weekend'

    # add in years group
    df.loc[(df['year'] < 2020) & (df['year'] >= 2017), 'year_group'] = '2017-2019'
    df.loc[df['year'] == 2020, 'year_group'] = '2020'

    # limit to covid period
    if do_limit_to_covid_dates:
        df = limit_to_dates(df, df['date'], min_date_str, max_date_str)

    if exclude_sensors is not None:
        df = df[~df['sensor_id'].isin(exclude_sensors)]

    # restrict to classes of interest to speed up
    if classes_of_interest is not None:
        df = df.drop([c for c in classes if c not in classes_of_interest], axis=1)
        classes = classes_of_interest

    id_cols = ['hour_of_day', 'date', 'datetime', 'year', 'weekday', 'sensor_id',
               'year_group', 'period']
    df = pd.melt(df, id_vars=id_cols,
                 value_vars=classes,
                 value_name='likelihood',
                 var_name='sound_class')

    df = restrict_to_spl_dates(df, SPL_DATAPATH, do_limit_to_covid_dates=False,
                          max_date_str=max_date_str, min_date_str=min_date_str)

    return df


def compute_presence(df, thresholds=None, id_cols=None):
    '''
    Function to compute the presence of the different sound class given their likelihood and an adequate threshold.
    The presence is averaged at the hour level.

    Parameters
    ----------
    df (pandas DataFrame):
        DataFrame containing the classes likelihood.
    thresholds (dict):
        Dictionary with customize thresholds, format is {sound_class:threshold_value}.
        By default the presence is computed with a threshold of 0.2.

    Returns
    -------

    '''

    df = df.copy()

    df['presence'] = (df['likelihood'] > 0.2).astype('float')

    if thresholds is not None:
        for sound_class in thresholds.keys():
            df.loc[df['sound_class'] == sound_class, 'presence'] = (df['likelihood'] >
                                                                   thresholds[sound_class]).astype('float')
    if id_cols is None:
        id_cols = ['sensor_id', 'year', 'weekday', 'date', 'hour_of_day', 'sound_class', 'period',
                        'year_group']
    df_sum = df.groupby(id_cols)['presence'].mean().reset_index()
    df_sum['datetime'] = df_sum['date'] + pd.Series([pd.DateOffset(hour=h) for h in df_sum['hour_of_day']])

    return df_sum


def restrict_to_spl_dates(df, SPL_DATAPATH, do_limit_to_covid_dates=False,
                          max_date_str='04-18', min_date_str='02-24'):
    '''
    Restrict the dates of the audio analysis to be the same as the SPL analysis.

    Parameters
    ----------
    df (pandas DataFrame):
        Dataframe with the audio analysis data
    SPL_DATAPATH (str):
        Path to spl.h5 file with SPL analysis data
    do_limit_to_covid_dates (bool):
        Wheter or not limit the analysis dates to a specific period
    max_date_str (str):
        Month and day of the upper day limit of the analysis period
    min_date_str (str):
        Month and day of the lower day limit of the analysis period

    Returns
    -------
    df (pandas DataFrame):
        DataFrame with restricted dates

    '''

    df = df.copy()

    df_spl = load_spl_data(SPL_DATAPATH,
                           do_limit_to_covid_dates=do_limit_to_covid_dates,
                           max_date_str=max_date_str, min_date_str=min_date_str)
    df = df[df['date'].isin(df_spl['date'])]

    return df


def presence_change_percentage(df, upper_bound=3, lower_bound=-3):
    '''
    Compute the percentage of change of previous presence with respect to historical template.

    Parameters
    ----------
    df_sum (pandas DataFrame):
        DataFrame to compute the change. The rolling_past_value and presence should be already computed.

    Returns
    -------
    df_sum (pandas DataFrame):
        DataFrame with computed change.

    '''

    df = df.copy()

    df['presence_change_per'] = (df['presence'] - df['rolling_past_value']) / \
                                (df['rolling_past_value'] + np.finfo(float).eps)
    df['presence_change_per'] = df['presence_change_per'].clip(upper=upper_bound, lower=lower_bound)
    df['presence_change_per'] = df['presence_change_per'] * 100

    return df

