import pandas as pd
import igraph
import configparser
import pycountry
import ray
from progress.bar import Bar
from progress.spinner import MoonSpinner
from matplotlib.colors import LinearSegmentedColormap


def load_config(config_file):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    return cfg


def get_config(cfg, section, key):
    try:
        return cfg[section][key]
    except KeyError:
        print(f'section {section} or key {key} not found')


G7 = ['United States', 'United Kingdom', 'Germany', 'France', 'Japan', 'Italy','Canada']
BRICS =  ['China', 'Brazil', 'India', 'South Africa', 'Russia']


def get_color(country):
    if country in G7:
        return 'green'
    elif country in BRICS:
        return 'orange'
    else:
        return 'lightyellow'


def get_country(name):
    try:
        code = pycountry.countries.get(name=name).alpha_2
        return code
    except:
        return name


def get_g7_brics_country(name):
    try:
        code = None
        if (name in G7) or (name in BRICS):
            code = pycountry.countries.get(name=name).alpha_2
        return code
    except:
        return name


def load_dataframe(data_file, ts_col, cols_to_load, end_timestep):
    df = pd.read_csv(data_file)
    df.index = df.index + 1
    df = df[cols_to_load]
    df[ts_col] = df.index
    ret_df = df[df[ts_col] < end_timestep]
    ret_df = filter_dataframe(ret_df, ts_col, 2)
    original_graph = igraph.Graph.DataFrame(ret_df, directed=False)
    tt = max(ret_df[ts_col])
    return ret_df, original_graph, tt


def filter_dataframe(df, ts_col, num_to_pick):
    drop_df = df.drop_duplicates(subset=['FromCty', 'ToCty'], keep='first')
    drop_df = drop_df.reset_index(drop=True)
    drop_df[ts_col] = drop_df.index
    grouped_df = drop_df.groupby(['FromCty'], as_index=False, sort=False).agg(lambda to: list(to))
    grouped_df = grouped_df[['FromCty', 'ToCty', 'timestep']]
    grouped_df['timestep'] = grouped_df.index
    grouped_df['ToCty'] = grouped_df['ToCty'].map(lambda x: x[0:num_to_pick])
    grouped_df = grouped_df.apply(pd.Series.explode).reset_index()

    ret_df = grouped_df[['FromCty', 'ToCty', 'timestep']]
    ret_df['timestep'] = ret_df.index

    print(f'filtered dataframe size is {len(ret_df.index)}')
    return ret_df


def delete_edges(g, t):
    g_copy = g.copy()
    g_copy.delete_edges(timestep_gt=t)
    return g_copy


def delete_vertices(g):
    g_copy = g.copy()
    to_delete_ids = [v.index for v in g_copy.vs if v.degree() == 0]
    g_copy.delete_vertices(to_delete_ids)
    return g_copy


cmap2 = LinearSegmentedColormap.from_list("edge_cmap", ["lightblue", "midnightblue"])


def get_size(graph):
    sz = []
    for vs in graph.vs:
        name = vs['name']
        if (name in G7) or (name in BRICS):
            btwn = vs.betweenness()
            print(f'betweeness {btwn} for {name}')
            if btwn:
                sz.append(igraph.rescale(vs.betweenness(), (50, 100)))
            else:
                sz.append(10.0)
        else:
            sz.append(10.0)
    return sz


def format_graph(graph, country_label: False):
    g_copy = graph.copy()

    if country_label:
        g_copy.vs['label'] = [get_country(name) for name in g_copy.vs['name']]

    g_copy.vs['color'] = [get_color(x) for x in g_copy.vs['name']]

#    g_copy.vs["size"] = get_size(g_copy)

    vs_betweenness = g_copy.betweenness()
    if vs_betweenness:
        g_copy.vs["size"]  = igraph.rescale(vs_betweenness, (30, 80))

    edge_betweenness = g_copy.edge_betweenness()
    if edge_betweenness:
        scaled_edge_betweenness = igraph.rescale(edge_betweenness, clamp=True)
        g_copy.es["color"] = [cmap2(betweenness) for betweenness in scaled_edge_betweenness]
        g_copy.es["width"] = igraph.rescale(edge_betweenness, (0.1, 1.0))

    return g_copy


def bucket_df(df, ts_col, bucket_sz):
    interval = pd.interval_range(0, df[ts_col].iat[-1], bucket_sz)
    bucketed = df.groupby(pd.cut(df[ts_col], bins=interval, right=True)).agg(lambda x: list(x))
    return bucketed


def construct_graph(original_graph, layout, bucketed_df, ts_col, country_label):
    old_layout = original_graph.layout(layout)
    n = 1
    with MoonSpinner('Generating graph...') as bar:
        for idx, row in bucketed_df.iterrows():
            last_ts_per_bucket = row[ts_col][-1]
            g = delete_edges(original_graph, last_ts_per_bucket)
            new_layout = g.layout(layout, seed=old_layout)
            gg = format_graph(delete_vertices(g), country_label)
            tgt = f'{output_folder}/example{n}.png'
            igraph.plot(gg, layout=new_layout, target=tgt, bbox=(1600,900))
            old_layout = new_layout.copy()
            n += 1
            bar.next()


def construct_filtered_graph(original_graph, layout, df, ts_col, max_ts, country_label):
#    old_layout = original_graph.layout(layout, niter=1000, start_temp=0.01, grid='nogrid')
    old_layout = original_graph.layout(layout)
    n = 1
    xs = (x / 10 for x in range(0, max_ts * 10))
#    xs = range(1, max_ts)
    for i in xs:
        g = delete_edges(original_graph, i)
        #new_layout = g.layout(layout, niter=1000, seed=old_layout, start_temp=0.01, grid='nogrid')
        new_layout = g.layout(layout, seed=old_layout)
        gg = format_graph(delete_vertices(g), country_label)
        n = int(i*10)
        #n = int(i*1)
        tgt = f'{output_folder}/example{n}.png'
        igraph.plot(gg, layout=new_layout, target=tgt, bbox=(1600,900))
        old_layout = new_layout.copy()
        n += 1



@ray.remote
def betweenness(original_graph, ts):
    print(f'generating betweenness for ts: {ts}')
    g = delete_vertices(delete_edges(original_graph, ts))
    df = pd.DataFrame([igraph.rescale(g.betweenness(), (0.1, 1))], columns=[v['name'] for v in g.vs], index=[ts])
    return df


def generate_betweenness_timeseries(original_graph, df, ts_col):
    ret_df = pd.DataFrame()
    results = ray.get([betweenness.remote(original_graph, ts) for ts in df[ts_col]])
    for i in results:
        print('done betweenness')
        ret_df = ret_df.append(i)
    return ret_df


def filter_betweenness(df, start_year):
    drop_df = df.drop_duplicates(subset=['FromCty', 'ToCty', 'Year'], keep='last')
    ret_df = drop_df.loc[drop_df['Year'] >= start_year]
    return ret_df


def degree_out(graph, ts):
    g = delete_vertices(delete_edges(graph, ts))
    deg = [v.degree(mode='out') for v in g.vs]
    df = pd.DataFrame([deg], columns=[v['name'] for v in g.vs], index=[ts])
    return df


def generate_degree_timeseries(original_graph, df, ts_col):
    ret_df = pd.DataFrame()
    results = [degree_out(original_graph, ts) for ts in df[ts_col]]
    for i in results:
        ret_df = ret_df.append(i)

    return ret_df

def plot_and_save(df, filename, col_name):
    df_plot = df.plot()
    df_plot.set_xlabel('timestep')
    df_plot.set_ylabel(col_name)
    df_fig = df_plot.get_figure()
    df_fig.savefig(filename)


cfg = load_config("config.ini")

data_file = get_config(cfg, 'Data', 'data_file')
output_folder = get_config(cfg, 'Data', 'output_folder')

ts_col = get_config(cfg, 'Simulation', 'timestep_col_name')
start_ts = int(get_config(cfg, 'Simulation', 'start_timestep'))
end_ts = int(get_config(cfg, 'Simulation', 'end_timestep'))
bucket_sz = int(get_config(cfg, 'Simulation', 'bucket_size'))
is_bucketed = True if get_config(cfg, 'Simulation', 'is_bucketed') == 'True' else False

country_label = True if cfg['Format']['country_label'] == 'True' else False
layout_name = cfg['Format']['layout_name']

str= f'''
Readin from : {data_file},
timestep col: {ts_col},
start_timestep: {start_ts},
end_timestep: {end_ts},
is_bucketed: {is_bucketed},
bucket_sz: {bucket_sz}, and 
writing to: {output_folder} with 
country_label: {country_label}
'''
print(str)

cols_to_load = ['FromCty', 'ToCty', 'Year', 'Quarter']
df, original_graph, total_time = load_dataframe(data_file, ts_col, cols_to_load, end_ts)

print(f'total simulation time is {total_time}')

run_animation = True if cfg['Workflow']['run_animation'] == 'True' else False
run_betweenness = True if cfg['Workflow']['run_betweenness'] == 'True' else False
run_degree = True if cfg['Workflow']['run_degree'] == 'True' else False


if run_animation:
    if is_bucketed:
        bucketed_df= bucket_df(df, ts_col, bucket_sz)
        construct_graph(original_graph, layout_name, bucketed_df, ts_col, country_label)
    else:
        construct_filtered_graph(original_graph, layout_name, df, ts_col, total_time, country_label)


if run_degree:
    filtered_df = filter_betweenness(df, 2018)
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['timestep']=filtered_df.index
    print(len(filtered_df))
    degree_out_df = generate_degree_timeseries(original_graph, filtered_df, ts_col)
    G7.append('China')
    g7_degree_out_df = degree_out_df[G7]
    g7_degree_out_df.to_csv('g7_degree.csv')
    out_file=f'{output_folder}/g7_degree.png'
    plot_and_save(g7_degree_out_df, out_file, 'degree')

    brics_degree_out_df = degree_out_df[BRICS]
    brics_degree_out_df.to_csv('brics_degree.csv')
    out_file=f'{output_folder}/brics_degree.png'
    plot_and_save(brics_degree_out_df, out_file, 'degree')

if run_betweenness:
    ray.init()
    filtered_df = filter_betweenness(df, 2018)
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['timestep']=filtered_df.index
    print(len(filtered_df))
    between_df = generate_betweenness_timeseries(original_graph, filtered_df, ts_col)
    between_df.to_csv('betweenness.csv')

    G7.append('China')
    g7_betweenness_df = between_df[G7]
    g7_betweenness_df.to_csv('g7_betweenness.csv')
    out_file=f'{output_folder}/g7.png'
    plot_and_save(g7_betweenness_df, out_file, 'betweenness')

    brics_betweenness_df = between_df[BRICS]
    brics_betweenness_df.to_csv('brics_betweenness.csv')
    out_file=f'{output_folder}/brics.png'
    plot_and_save(brics_betweenness_df, f'{output_folder}/brics.png')
