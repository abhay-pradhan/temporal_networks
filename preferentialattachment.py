import pandas as pd
import igraph
import configparser


def load_dataframe(data_file, df_col, end_timestep):
    df = pd.read_csv(data_file)[lambda x: x[df_col] < end_timestep]
    tt = max(df[df_col])
    original_graph = igraph.Graph.DataFrame(df, directed=False, use_vids=True)
    return df, original_graph, tt


# makes a copy of the graph, deletes edges greater than time t and returns the graph
def delete_edges(g, t):
    g_copy = g.copy()
    g_copy.delete_edges(time_gt=t)
    return g_copy


# makes a copy of the graph, deletes vertices with no edges and returns the graph
def delete_vertices(g, t):
    g_copy = g.copy()
    to_delete_ids = [v.index for v in g_copy.vs if v.degree() == 0]
    g_copy.delete_vertices(to_delete_ids)
    return g_copy


def load_config(config_file):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    return cfg


def get_config(cfg, section, key):
    try:
        return cfg[section][key]
    except KeyError:
        print(f'section {section} or key {key} not found')


def bucket_df(df, df_col, bucket_sz):
    interval = pd.interval_range(0, df[df_col].iat[-1], bucket_sz)
    bucketed = df.groupby(pd.cut(df[df_col], bins=interval, right=True)).agg(lambda x: list(x))
    return bucketed


def construct_graph(original_graph, bucketed_df, df_col):
    old_layout = original_graph.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid')
    n = 1
    for idx, row in bucketed_df.iterrows():
        last_ts_per_bucket = row[df_col][-1]
        g = delete_edges(original_graph, last_ts_per_bucket)
        new_layout = g.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid', seed=old_layout)
        gg = delete_vertices(g, last_ts_per_bucket)
        tgt = f'{output_folder}/example{n}.png'
        print(tgt)
        igraph.plot(gg, layout=new_layout, target=tgt, bbox=(1600,900))
        old_layout = new_layout.copy()
        n += 1


# load configuration from a config.ini file present in current directory
cfg = load_config("config.ini")

data_file = get_config(cfg, 'Data', 'data_file')
output_folder = get_config(cfg, 'Data', 'output_folder')

df_col = get_config(cfg, 'Simulation', 'timestep_col_name')
start_ts = int(get_config(cfg, 'Simulation', 'start_timestep'))
end_ts = int(get_config(cfg, 'Simulation', 'end_timestep'))
bucket_sz = int(get_config(cfg, 'Simulation', 'bucket_size'))

print(f'Reading from {data_file} with timestep column name {df_col} start_timestep {start_ts}, end_timestep {end_ts} and bucket_size {bucket_sz} and writing to output_folder {output_folder}')
df, original_graph, total_time = load_dataframe(data_file, df_col, end_ts)
print(f'total simulation time is {total_time}')

bucketed_df= bucket_df(df, df_col, bucket_sz)
construct_graph(original_graph, bucketed_df, df_col)
